"""
LoRA 实现

原理:
  冻结预训练权重 W，在目标线性层旁添加低秩分解分支:
    h = Wx + ΔWx = Wx + (α/r) * B * A * x
  其中 A ∈ R^{r×in}, B ∈ R^{out×r}, r << min(in, out)
  训练时只更新 A 和 B，大幅减少可训练参数量。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearWithLoRA(nn.Module):
  """
  在线性层上添加 LoRA 分支。
  原始权重被冻结，仅训练低秩矩阵 lora_A 和 lora_B。
  """

  def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.1):
    super().__init__()
    self.original_linear = original_linear
    self.r = r
    self.alpha = alpha
    self.scaling = alpha / r

    in_features = original_linear.in_features
    out_features = original_linear.out_features

    # 冻结原始权重
    self.original_linear.weight.requires_grad = False
    if self.original_linear.bias is not None:
      self.original_linear.bias.requires_grad = False

    # 低秩分解矩阵
    # lora_A: (r, in_features) = (8, 768) — 用 Kaiming 初始化
    self.lora_A = nn.Parameter(torch.zeros(r, in_features))
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    # lora_B: (out_features, r) = (768, 8) — 初始化为 0，保证训练开始时仍然是 h = Wx
    self.lora_B = nn.Parameter(torch.zeros(out_features, r))
    nn.init.zeros_(self.lora_B)

    self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 原始路径，即 result = Wx + b
    result = self.original_linear(x)
    # LoRA 路径: x -> dropout -> A -> B -> scaling
    # 首先 dropout，x_dropped = self.lora_dropout(x)   [32, 128, 768]
    # 通过 A 降维 low_dim = F.linear(x_dropped, self.lora_A)
    # 再通过 B 升维 lora_out = F.linear(low_dim, self.lora_B)
    # 应用 scaling、合并 output = result + self.scaling * lora_out
    lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
    result = result + self.scaling * lora_out
    return result

  @torch.no_grad()
  def merge_weights(self):
    """
    推理时将 LoRA 权重合并回原始权重，消除额外计算开销。
    W_merged = W + (α/r) * B @ A
    """
    # 将 lora 权重合并回原始的权重中，得到新的矩阵 W_merged = W + (α/r) * B @ A
    self.original_linear.weight.data += self.scaling * (self.lora_B @ self.lora_A)

  @torch.no_grad()
  def unmerge_weights(self):
    """撤销合并，恢复原始权重。"""
    # 在我们跑 training 的时候，我们需要冻结 W，这时只计算对 A, B 的梯度并更新，所以不能合并
    # 在推理的时候可以合并，此时模型在推理时的 computation graph、计算量、显存占用和原始的 
    # GPT-2 相同，没有延迟
    self.original_linear.weight.data -= self.scaling * (self.lora_B @ self.lora_A)


def apply_lora_to_gpt2(model, r: int = 8, alpha: float = 16.0, dropout: float = 0.1,
                        target_modules: list = None):
  """
  对 GPT-2 模型的目标线性层应用 LoRA

  Args:
    model: GPT2Model 实例
    r: LoRA 秩
    alpha: LoRA 缩放因子
    dropout: LoRA dropout 概率
    target_modules: 要替换的模块名称列表，如 ['query', 'value']
                    默认为 ['query', 'value']（遵循原论文）

  Returns:
    lora_params: 仅包含 LoRA 参数的列表（用于优化器）
  """
  if target_modules is None:
    # 默认处理 Q 和 V，这也是 LoRA 论文提出来的，K 相对鸡肋一些
    target_modules = ['query', 'value']

  lora_params = []
  replaced_count = 0

  # 遍历所有 GPT-2 层
  for gpt_layer in model.gpt_layers:
    attn = gpt_layer.self_attention

    for module_name in target_modules:
      if not hasattr(attn, module_name):
        continue

      
      original_linear = getattr(attn, module_name)
      if not isinstance(original_linear, nn.Linear):
        continue

      # 将原始的线性层替换为 LoRA 层
      lora_linear = LinearWithLoRA(original_linear, r=r, alpha=alpha, dropout=dropout)
      setattr(attn, module_name, lora_linear)

      # 收集 LoRA 参数
      lora_params.extend([lora_linear.lora_A, lora_linear.lora_B])
      replaced_count += 1

  print(f"[LoRA] Applied LoRA to {replaced_count} modules across {len(model.gpt_layers)} layers")
  print(f"[LoRA] Target modules: {target_modules}, r={r}, alpha={alpha}")

  return lora_params


def freeze_base_model(model):
  """冻结 GPT-2 基础模型的所有参数。"""
  frozen_count = 0
  for param in model.parameters():
    param.requires_grad = False
    frozen_count += 1
  print(f"[LoRA] Froze {frozen_count} base model parameters")


def get_lora_parameter_stats(model):
  """获取 LoRA 参数统计信息。"""
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  frozen_params = total_params - trainable_params

  stats = {
    "total_params": total_params,
    "trainable_params": trainable_params,
    "frozen_params": frozen_params,
    "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
  }

  print(f"[LoRA] Parameter statistics:")
  print(f"  Total:     {total_params:>12,}")
  print(f"  Trainable: {trainable_params:>12,} ({stats['trainable_ratio']*100:.2f}%)")
  print(f"  Frozen:    {frozen_params:>12,} ({(1-stats['trainable_ratio'])*100:.2f}%)")

  return stats
