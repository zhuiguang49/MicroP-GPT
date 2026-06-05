'''
Paraphrase detection with LoRA (Low-Rank Adaptation).

基于 paraphrase_detection.py，使用 LoRA 进行参数高效微调。
仅训练 LoRA 分支和分类头，冻结 GPT-2 基础模型。

核心思想：
=========
1. 冻结预训练的 GPT-2 模型（~124M 参数）
2. 在注意力层的 query 和 value 投影上添加 LoRA 适配器（~0.5M 参数）
3. 只训练 LoRA 参数 + 分类头参数
4. 大幅减少显存占用和训练时间，同时保持良好性能

Running:
  `python paraphrase_detection_lora.py --use_gpu --lora_r 8 --lora_alpha 16`
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from logging_utils import ExperimentLogger
from lora import apply_lora_to_gpt2, freeze_base_model, get_lora_parameter_stats

from optimizer import AdamW

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPTLoRA(nn.Module):
  """
  使用 LoRA 的 GPT-2 Paraphrase Detection 模型。
  
  架构设计：
  ---------
  输入句子对 → GPT-2 (冻结 + LoRA) → 最后一个 token 表示 → 分类头 → 是否同义
  
  关键组件：
  1. GPT-2 编码器：提取句子对的语义表示（基础权重冻结，通过 LoRA 适配）
  2. LoRA 适配器：在 attention 的 Q/V 投影上添加低秩分支（可训练）
  3. 分类头：将最后一个 token 的隐藏状态映射到二分类 logits（可训练）
  
  数据流示例：
  ----------
  输入："How to learn Python?" + "What's the best way to study Python?"
    ↓
  Tokenization: [CLS] How to learn Python ? [SEP] What ' s the best ... [SEP]
    ↓
  GPT-2 + LoRA: 生成每个 token 的 hidden state [batch, seq_len, 768]
    ↓
  提取最后一个非 padding token: [batch, 768]
    ↓
  分类头 (Linear 768→2): [batch, 2] (logits for [no, yes])
    ↓
  输出：是否是同义句的概率分布
  """

  def __init__(self, args):
    """
    初始化模型
    
    Args:
        args: 命令行参数，包含：
            - model_size: GPT-2 模型大小 ('gpt2', 'gpt2-medium', 'gpt2-large')
            - d: 隐藏层维度 (768/1024/1280)
            - l: Transformer 层数 (12/24/36)
            - num_heads: 注意力头数 (12/16/20)
            - lora_r: LoRA 秩
            - lora_alpha: LoRA 缩放因子
            - lora_dropout: LoRA dropout 概率
            - lora_target_modules: 目标模块列表（如 'query,value'）
    """
    super().__init__()
    
    self.gpt = GPT2Model.from_pretrained(
      model=args.model_size, 
      d=args.d, 
      l=args.l, 
      num_heads=args.num_heads
    )
    
    self.paraphrase_detection_head = nn.Linear(args.d, 2)

    freeze_base_model(self.gpt)

    # ========================================
    # 对 attention 层应用 LoRA
    # ========================================
    # 解析目标模块列表（默认是 query 和 value）
    target_modules = args.lora_target_modules.split(',') if args.lora_target_modules else ['query', 'value']
    
    # apply_lora_to_gpt2 会：
    # 1. 遍历所有 GPT-2 层的 self_attention 模块
    # 2. 将 target_modules 指定的线性层替换为 LinearWithLoRA
    # 3. 返回所有 LoRA 参数（lora_A 和 lora_B）
    self.lora_params = apply_lora_to_gpt2(
      self.gpt,
      r=args.lora_r,           # LoRA 秩，控制参数量
      alpha=args.lora_alpha,   # 缩放因子，控制学习强度
      dropout=args.lora_dropout,  # dropout 概率，正则化
      target_modules=target_modules  # 要应用 LoRA 的模块
    )
    
    # 此时模型状态：
    # - GPT-2 基础权重：冻结（requires_grad=False）
    # - LoRA 参数：可训练（requires_grad=True）
    # - 分类头：可训练（但当前未使用）

  def forward(self, input_ids, attention_mask):
    """
    前向传播：使用 cloze-style 方法进行同义句检测
    
    Cloze-style 方法原理：
    --------------------
    传统分类：最后一个 token → Linear → [no_logits, yes_logits]
    Cloze-style：让模型直接预测 "yes" 或 "no" token 的概率
  
    
    Args:
        input_ids: 输入 token IDs，形状 [batch_size, seq_len]
                  例如：[[101, 2054, 2000, ...], [...]]
        attention_mask: 注意力掩码，形状 [batch_size, seq_len]
                       1 表示真实 token，0 表示 padding
    
    Returns:
        logits: 分类 logits，形状 [batch_size, 2]
               第 0 列是 "no" 的 logit，第 1 列是 "yes" 的 logit
    """
    # 步骤 1: 通过 GPT-2 编码器获取隐藏状态
    # outputs 是一个字典，包含：
    # - 'last_hidden_state': 所有 token 的隐藏状态 [batch, seq_len, hidden_size]
    # - 'last_token': 最后一个非 padding token 的隐藏状态 [batch, hidden_size]
    outputs = self.gpt(input_ids, attention_mask)
    
    # 步骤 2: 提取最后一个 token 的表示
    last_token_hidden = outputs['last_token']  # [batch_size, hidden_size]
    
    # 步骤 3: 将 hidden state 投影到词汇表空间
    # hidden_state_to_token 执行：logits = hidden_state @ embedding_weight.T
    # 这是 GPT-2 的 weight tying 技术，输出层权重与输入 embedding 共享
    # 输出形状：[batch_size, vocab_size] (vocab_size ≈ 50257 for GPT-2)
    vocab_logits = self.gpt.hidden_state_to_token(last_token_hidden)
    
    # 步骤 4: 提取 "yes" 和 "no" token 的 logits
    # GPT-2 词汇表中：
    # - token_id 8505 对应 "yes"
    # - token_id 3919 对应 "no"
    # 这些 ID 是通过查看 GPT-2 tokenizer 的词汇表确定的
    yes_logits = vocab_logits[:, 8505].unsqueeze(1)  # [batch_size, 1]
    no_logits = vocab_logits[:, 3919].unsqueeze(1)   # [batch_size, 1]
    
    # 步骤 5: 拼接成二分类 logits
    return torch.cat([no_logits, yes_logits], dim=1)  # [batch_size, 2]

  def get_trainable_params(self):
    """
    返回所有可训练参数（LoRA 参数 + 分类头参数）。
    
    Returns:
        trainable: 参数列表，包含：
          - 所有 LoRA 层的 lora_A 和 lora_B
          - 分类头的 weight 和 bias（如果使用）
    """
    # 合并 LoRA 参数和分类头参数
    trainable = list(self.lora_params) + list(self.paraphrase_detection_head.parameters())
    return trainable


def save_model(model, optimizer, args, filepath):
  """
  保存模型、优化器状态和随机种子，用于后续恢复或测试。
  
  Args:
      model: 要保存的模型实例
      optimizer: 优化器实例
      args: 命令行参数
      filepath: 保存路径（.pt 文件）
  """
  save_info = {
    'model': model.state_dict(),      # 模型参数
    'optim': optimizer.state_dict(),  # 优化器状态
    'args': args,                     # 超参数
    'system_rng': random.getstate(),  # Python 随机数状态
    'numpy_rng': np.random.get_state(),  # NumPy 随机数状态
    'torch_rng': torch.random.get_rng_state(),  # PyTorch 随机数状态
  }
  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """
  使用 LoRA 训练 GPT-2 进行 Quora 数据集的同义句检测。
  
  训练流程：
  ---------
  1. 加载数据（训练集 + 验证集）
  2. 创建数据加载器（DataLoader）
  3. 初始化模型（GPT-2 + LoRA）
  4. 创建优化器（仅优化 LoRA 参数）
  5. 多轮训练：
     - 前向传播计算 loss
     - 反向传播计算梯度
     - 更新 LoRA 参数
     - 在验证集上评估
  6. 保存最佳模型
  

  - 小学习率（1e-5）
  - 小 batch size（8）
  - 较多 epoch（10）
  """

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')


  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)


  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(
    para_train_data, 
    shuffle=True, 
    batch_size=args.batch_size,
    collate_fn=para_train_data.collate_fn
  )
  para_dev_dataloader = DataLoader(
    para_dev_data, 
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=para_dev_data.collate_fn
  )

  args = add_arguments(args)
  
  # 创建模型实例
  model = ParaphraseGPTLoRA(args)
  model = model.to(device)  

  # 创建优化器：仅优化可训练参数（LoRA + 分类头）
  # lr=1e-5: 小学习率，因为：
  #   1. 只更新少量参数，大步长容易震荡
  #   2. 保护预训练知识不被破坏
  lr = args.lr
  optimizer = AdamW(model.get_trainable_params(), lr=lr, weight_decay=0.)
  
  best_dev_acc = 0  # 记录最佳验证准确率


  lora_stats = get_lora_parameter_stats(model)

  logger = ExperimentLogger(
    experiment_name=args.exp_name, 
    task="paraphrase",             
    method="lora",                 
    output_dir="logs"           
  )
  
  logger.log_config(vars(args))  
  logger.log_model_info(model, method_specific_info={
    "lora_r": args.lora_r,
    "lora_alpha": args.lora_alpha,
    "lora_dropout": args.lora_dropout,
    "lora_target_modules": args.lora_target_modules,
    **lora_stats, 
  })
  logger.log_training_start()


  for epoch in range(args.epochs):
    logger.log_epoch_start()
    model.train()  
    train_loss = 0  
    num_batches = 0  


    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      labels = (labels == 8505).long()


      optimizer.zero_grad()
      
      # 前向传播
      # 输入：token IDs 和 attention mask
      # 输出：logits [batch_size, 2]，分别是 "no" 和 "yes" 的得分
      logits = model(b_ids, b_mask)
      
      # 计算损失
      # F.cross_entropy 内部会：
      #   1. 对 logits 应用 softmax 得到概率
      #   2. 计算交叉熵损失 -log(p_true_label)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      
      # 反向传播
      # 计算所有可训练参数（LoRA + 分类头）的梯度
      loss.backward()
      
      # 更新参数
      # AdamW 根据梯度和动量更新 LoRA 参数
      optimizer.step()

      # 累计损失（用于计算平均损失）
      train_loss += loss.item()
      num_batches += 1

    # 计算平均训练损失
    train_loss = train_loss / num_batches
    
    # 在验证集上评估
    # model_eval_paraphrase 返回：准确率、F1、预测标签、真实标签、样本 IDs
    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")

    logger.log_epoch_metrics(epoch, {
      "train_loss": train_loss,
      "dev_acc": dev_acc,
      "dev_f1": dev_f1,
      "best_dev_acc": best_dev_acc,
    })


  logger.log_training_end()
  logger.log_final_results({
    "best_dev_acc": best_dev_acc,
    "final_dev_f1": dev_f1,
  })
  logger.save()
  logger.print_summary()


@torch.no_grad()
def test(args):
  """
  在验证集和测试集上评估模型，保存预测结果到磁盘。
  
  测试流程：
  1. 加载保存的最佳模型
  2. 加载验证集和测试集数据
  3. 在两个数据集上进行推理
  4. 将预测结果保存为 CSV 文件
  
  """

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  saved = torch.load(args.filepath, weights_only=False)

  model = ParaphraseGPTLoRA(saved['args'])
  
  model.load_state_dict(saved['model'])

  model = model.to(device)
  model.eval()  
  print(f"Loaded model to test from {args.filepath}")


  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)


  para_dev_dataloader = DataLoader(
    para_dev_data, 
    shuffle=False, 
    batch_size=args.batch_size,
    collate_fn=para_dev_data.collate_fn
  )
  para_test_dataloader = DataLoader(
    para_test_data, 
    shuffle=True, 
    batch_size=args.batch_size,
    collate_fn=para_test_data.collate_fn
  )

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(
    para_dev_dataloader, model, device
  )
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  

  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(
    para_test_dataloader, model, device
  )


  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")  
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n") 

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv",
                      help="训练集 CSV 文件路径")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv",
                      help="验证集 CSV 文件路径")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv",
                      help="测试集 CSV 文件路径")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output-lora.csv",
                      help="验证集预测结果输出路径")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output-lora.csv",
                      help="测试集预测结果输出路径")

  parser.add_argument("--seed", type=int, default=11711,
                      help="随机种子，确保可复现性")
  parser.add_argument("--epochs", type=int, default=10,
                      help="训练轮数")
  parser.add_argument("--use_gpu", action='store_true',
                      help="是否使用 GPU 加速")

  parser.add_argument("--batch_size", type=int, default=8,
                      help="批大小（LoRA 可以用较小的 batch size）")
  parser.add_argument("--lr", type=float, default=1e-5,
                      help="学习率（LoRA 通常使用较小的学习率）")
  parser.add_argument("--model_size", type=str,
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], 
                      default='gpt2',
                      help="GPT-2 模型大小")
  parser.add_argument("--exp_name", type=str, default="lora_r8",
                      help="实验名称，用于日志记录")

  parser.add_argument("--lora_r", type=int, default=8, 
                      help="LoRA 秩，控制低秩矩阵的维度。"
                           "典型值：4, 8, 16, 32。"
                           "r 越小参数越少，但表达能力受限；"
                           "r 越大参数越多，可能过拟合。")
  
  parser.add_argument("--lora_alpha", type=float, default=16.0, 
                      help="LoRA 缩放因子，控制 LoRA 分支的学习强度。"
                           "经验法则：alpha = 2 * r 或 alpha = r。"
                           "scaling = alpha / r，实际应用到输出的系数。")
  
  parser.add_argument("--lora_dropout", type=float, default=0.1, 
                      help="LoRA 分支的 dropout 概率，防止过拟合。"
                           "典型值：0.05, 0.1, 0.15。"
                           "仅应用于 LoRA 路径，不影响原始权重。")
  
  parser.add_argument("--lora_target_modules", type=str, default="query,value",
                      help="要应用 LoRA 的目标模块，逗号分隔。"
                           "可选值：query, key, value, out_proj, interm_dense, out_dense 等。"
                           "默认 'query,value' 遵循原论文最佳实践。"
                           "示例：'query,key,value' 会对三个投影都应用 LoRA。")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """
  根据模型大小自动添加模型架构参数。
  
  GPT-2 系列模型的配置：
  ---------------------
  | 模型         | 隐藏维度(d) | 层数(l) | 注意力头数 |
  |-------------|-----------|--------|----------|
  | gpt2        | 768       | 12     | 12       |
  | gpt2-medium | 1024      | 24     | 16       |
  | gpt2-large  | 1280      | 36     | 20       |
  
  Args:
      args: 包含 model_size 的参数对象
  
  Returns:
      args: 添加了 d, l, num_heads 的参数对象
  """
  if args.model_size == 'gpt2':
    args.d = 768      # 隐藏层维度
    args.l = 12       # Transformer 层数
    args.num_heads = 12  # 注意力头数
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  """
  示例运行命令：
  -------------
  python paraphrase_detection_lora.py \
    --use_gpu \
    --lora_r 8 \
    --lora_alpha 16 \
    --batch_size 8 \
    --lr 1e-5 \
    --epochs 10 \
    --model_size gpt2
  """
 
  args = get_args()
  
  args.filepath = f'{args.epochs}-{args.lr}-lora_r{args.lora_r}-paraphrase.pt'
  
  seed_everything(args.seed)
  
  train(args)
  
  test(args)
