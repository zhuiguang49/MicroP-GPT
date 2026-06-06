'''
Sonnet generation with DPO (Direct Preference Optimization).

论文: Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)

DPO Loss:
  L = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))

训练流程:
  1. 加载 paired data (chosen = 真实 sonnet, rejected = GPT-2 生成的低质量补全)
  2. Policy model 从预训练权重初始化（或从 SFT checkpoint 初始化）
  3. Reference model 冻结为预训练 GPT-2
  4. 用 DPO loss 训练 policy model

Running:
  `python sonnet_generation_dpo.py --use_gpu --beta 0.1`
'''

import argparse
import copy
import json
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from logging_utils import ExperimentLogger
from sonnet_generation import SonnetGPT, add_arguments, save_model, seed_everything

from optimizer import AdamW

TQDM_DISABLE = False


# =============================================================================
# DPO Dataset
# =============================================================================

class DPOSonnetDataset(Dataset):
  """
  DPO 配对数据集
  每条数据: (prompt, chosen, rejected)
  """

  def __init__(self, json_path: str, tokenizer, max_length: int = 256):
    with open(json_path, 'r', encoding='utf-8') as f:
      self.data = json.load(f)
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    prompt = item['prompt']
    chosen = item['chosen']
    rejected = item['rejected']

    # Tokenize full sequences (prompt + completion)
    chosen_enc = self.tokenizer(
      chosen, return_tensors='pt', padding='max_length',
      truncation=True, max_length=self.max_length
    )
    rejected_enc = self.tokenizer(
      rejected, return_tensors='pt', padding='max_length',
      truncation=True, max_length=self.max_length
    )

    # 计算 prompt 的 token 数量（用于构建 completion mask）
    prompt_enc = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_length)
    prompt_len = prompt_enc['input_ids'].shape[1]

    return {
      'chosen_ids': chosen_enc['input_ids'].squeeze(0),
      'chosen_mask': chosen_enc['attention_mask'].squeeze(0),
      'rejected_ids': rejected_enc['input_ids'].squeeze(0),
      'rejected_mask': rejected_enc['attention_mask'].squeeze(0),
      'prompt_len': prompt_len,
    }

  @staticmethod
  def collate_fn(batch):
    return {
      'chosen_ids': torch.stack([x['chosen_ids'] for x in batch]),
      'chosen_mask': torch.stack([x['chosen_mask'] for x in batch]),
      'rejected_ids': torch.stack([x['rejected_ids'] for x in batch]),
      'rejected_mask': torch.stack([x['rejected_mask'] for x in batch]),
      'prompt_lens': torch.tensor([x['prompt_len'] for x in batch]),
    }


# =============================================================================
# DPO Core Functions
# =============================================================================

def compute_log_probs(model, input_ids, attention_mask):
  """
  计算模型对序列中每个 token 的 log probability。

  Args:
    model: GPT-2 模型
    input_ids: (batch, seq_len)
    attention_mask: (batch, seq_len)

  Returns:
    log_probs: (batch, seq_len-1) 每个预测位置的 log prob
  """
  # SonnetGPT.forward() 直接返回 (batch, seq_len, vocab_size) 的 logits
  logits = model(input_ids, attention_mask)  # (batch, seq_len, vocab_size)

  # Shift: 用位置 0..T-2 的 logits 预测位置 1..T-1 的 token
  shift_logits = logits[:, :-1, :].contiguous()
  shift_labels = input_ids[:, 1:].contiguous()

  # Per-token log probs
  log_probs = F.log_softmax(shift_logits, dim=-1)
  token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

  return token_log_probs


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta: float):
  """
  计算 DPO loss。

  Args:
    policy_chosen_logps: (batch,) policy model 对 chosen 序列的 log prob
    policy_rejected_logps: (batch,) policy model 对 rejected 序列的 log prob
    ref_chosen_logps: (batch,) reference model 对 chosen 序列的 log prob
    ref_rejected_logps: (batch,) reference model 对 rejected 序列的 log prob
    beta: KL 惩罚系数

  Returns:
    loss: scalar
    chosen_rewards: (batch,) chosen 样本的 reward
    rejected_rewards: (batch,) rejected 样本的 reward
  """
  # π_θ(y|x) / π_ref(y|x) 的对数差
  chosen_logratios = policy_chosen_logps - ref_chosen_logps
  rejected_logratios = policy_rejected_logps - ref_rejected_logps

  # DPO logits: β * (log ratio_chosen - log ratio_rejected)
  logits = beta * (chosen_logratios - rejected_logratios)

  # Loss: -log σ(logits)
  loss = -F.logsigmoid(logits).mean()

  # Rewards (用于日志)
  chosen_rewards = beta * chosen_logratios.detach()
  rejected_rewards = beta * rejected_logratios.detach()

  return loss, chosen_rewards, rejected_rewards


def get_sequence_log_probs(log_probs, attention_mask, prompt_lens):
  """
  对 completion 部分的 token log probs 求和，得到序列级别的 log prob。

  Args:
    log_probs: (batch, seq_len-1) per-token log probs
    attention_mask: (batch, seq_len) 原始 attention mask
    prompt_lens: (batch,) 每个样本的 prompt token 数量

  Returns:
    seq_log_probs: (batch,) 仅包含 completion 部分的 log prob 之和
  """
  batch_size, seq_len_minus_1 = log_probs.shape

  # 构建 completion mask: 在 shifted 空间中，位置 i 预测 token i+1
  # 如果 token i+1 属于 completion（即 i+1 >= prompt_len），则 mask=1
  # 等价于 i >= prompt_len - 1
  shifted_mask = attention_mask[:, 1:]  # (batch, seq_len-1)
  indices = torch.arange(seq_len_minus_1, device=log_probs.device).unsqueeze(0)
  completion_mask = (indices >= (prompt_lens - 1).unsqueeze(1)).float()
  completion_mask = completion_mask * shifted_mask  # 同时排除 padding

  # 仅对 completion 部分求和
  seq_log_probs = (log_probs * completion_mask).sum(dim=1)
  return seq_log_probs


# =============================================================================
# Training
# =============================================================================

def train(args):
  """Train GPT-2 with DPO for sonnet generation."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  args = add_arguments(args)
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token

  # ---- 数据 ----
  dpo_dataset = DPOSonnetDataset(args.dpo_data_path, tokenizer, max_length=args.max_length)
  dpo_dataloader = DataLoader(
    dpo_dataset, shuffle=True, batch_size=args.batch_size,
    collate_fn=DPOSonnetDataset.collate_fn
  )

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  # ---- Policy Model (可训练) ----
  policy_model = SonnetGPT(args)
  if args.sft_checkpoint:
    print(f"Loading SFT checkpoint from {args.sft_checkpoint}")
    saved = torch.load(args.sft_checkpoint, weights_only=False)
    policy_model.load_state_dict(saved['model'])
  policy_model = policy_model.to(device)

  # ---- Reference Model (冻结) ----
  ref_model = SonnetGPT(args)
  ref_model = ref_model.to(device)
  ref_model.eval()
  for param in ref_model.parameters():
    param.requires_grad = False

  # ---- 优化器 ----
  optimizer = AdamW(policy_model.parameters(), lr=args.lr)

  # ---- 实验日志 ----
  logger = ExperimentLogger(
    experiment_name=args.exp_name,
    task="sonnet",
    method="dpo",
    output_dir="logs"
  )
  logger.log_config(vars(args))
  logger.log_model_info(policy_model, method_specific_info={
    "beta": args.beta,
    "reference_model": "frozen_pretrained",
    "sft_checkpoint": args.sft_checkpoint or "none",
  })
  logger.log_training_start()

  # ---- 训练循环 ----
  best_chrF = 0

  for epoch in range(args.epochs):
    logger.log_epoch_start()
    policy_model.train()

    epoch_dpo_loss = 0
    epoch_chosen_reward = 0
    epoch_rejected_reward = 0
    num_batches = 0

    for batch in tqdm(dpo_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      chosen_ids = batch['chosen_ids'].to(device)
      chosen_mask = batch['chosen_mask'].to(device)
      rejected_ids = batch['rejected_ids'].to(device)
      rejected_mask = batch['rejected_mask'].to(device)
      prompt_lens = batch['prompt_lens'].to(device)

      # 1. 计算 policy model 的 log probs
      policy_chosen_lp = compute_log_probs(policy_model, chosen_ids, chosen_mask)
      policy_rejected_lp = compute_log_probs(policy_model, rejected_ids, rejected_mask)

      policy_chosen_seq_lp = get_sequence_log_probs(policy_chosen_lp, chosen_mask, prompt_lens)
      policy_rejected_seq_lp = get_sequence_log_probs(policy_rejected_lp, rejected_mask, prompt_lens)

      # 2. 计算 reference model 的 log probs (no grad)
      with torch.no_grad():
        ref_chosen_lp = compute_log_probs(ref_model, chosen_ids, chosen_mask)
        ref_rejected_lp = compute_log_probs(ref_model, rejected_ids, rejected_mask)

        ref_chosen_seq_lp = get_sequence_log_probs(ref_chosen_lp, chosen_mask, prompt_lens)
        ref_rejected_seq_lp = get_sequence_log_probs(ref_rejected_lp, rejected_mask, prompt_lens)

      # 3. DPO loss
      loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_seq_lp, policy_rejected_seq_lp,
        ref_chosen_seq_lp, ref_rejected_seq_lp,
        beta=args.beta
      )

      # 4. 反向传播
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_dpo_loss += loss.item()
      epoch_chosen_reward += chosen_rewards.mean().item()
      epoch_rejected_reward += rejected_rewards.mean().item()
      num_batches += 1

    # Epoch 统计
    avg_dpo_loss = epoch_dpo_loss / num_batches
    avg_chosen_reward = epoch_chosen_reward / num_batches
    avg_rejected_reward = epoch_rejected_reward / num_batches
    reward_margin = avg_chosen_reward - avg_rejected_reward

    print(f"Epoch {epoch}: dpo_loss :: {avg_dpo_loss :.4f}, "
          f"chosen_reward :: {avg_chosen_reward :.4f}, "
          f"rejected_reward :: {avg_rejected_reward :.4f}, "
          f"margin :: {reward_margin :.4f}")

    # ---- Dev 评估: 生成 sonnets 并计算 chrF ----
    print('Generating dev sonnets...')
    policy_model.eval()
    generated_sonnets = []

    for batch in held_out_sonnet_dataset:
      sonnet_id = batch[0]
      prompt = batch[1]
      encoding = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
      output = policy_model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      generated_sonnets.append((sonnet_id, output[1]))

    temp_dev_path = f'predictions/generated_sonnets_dpo_dev_epoch_{epoch}.txt'
    with open(temp_dev_path, 'w') as f:
      f.write('--Generated Sonnets--\n\n')
      for sonnet_id, sonnet_text in generated_sonnets:
        f.write(f'\n{sonnet_id}\n')
        f.write(sonnet_text)

    from evaluation import test_sonnet
    chrf_score = test_sonnet(
      test_path=temp_dev_path,
      gold_path='data/TRUE_sonnets_held_out_dev.txt'
    )
    print(f"Epoch {epoch}: dev chrF :: {chrf_score :.3f}")

    if chrf_score > best_chrF:
      best_chrF = chrf_score
      save_model(policy_model, optimizer, args, args.filepath)

    logger.log_epoch_metrics(epoch, {
      "dpo_loss": avg_dpo_loss,
      "chosen_reward": avg_chosen_reward,
      "rejected_reward": avg_rejected_reward,
      "reward_margin": reward_margin,
      "dev_chrF": chrf_score,
      "best_chrF": best_chrF,
    })

  # 训练结束
  logger.log_training_end()
  logger.log_final_results({
    "best_chrF": best_chrF,
    "final_dpo_loss": avg_dpo_loss,
    "final_reward_margin": reward_margin,
  })
  logger.save()
  logger.print_summary()
  save_model(policy_model, optimizer, args, f'{args.epochs}_{args.filepath}')


def get_args():
  parser = argparse.ArgumentParser()

  # 数据路径
  parser.add_argument("--dpo_data_path", type=str, default="data/sonnets_rejected.json",
                      help="Path to paired data JSON from generate_rejected_sonnets.py")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dpo.txt")

  # SFT checkpoint (可选)
  parser.add_argument("--sft_checkpoint", type=str, default=None,
                      help="Path to SFT fine-tuned checkpoint. If None, starts from pretrained GPT-2.")

  # 训练参数
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", type=int, default=4,
                      help="DPO 需要同时处理 chosen 和 rejected，显存占用更大")
  parser.add_argument("--lr", type=float, default=5e-6,
                      help="DPO 学习率通常比 SFT 更小")
  parser.add_argument("--max_length", type=int, default=256)

  # DPO 参数
  parser.add_argument("--beta", type=float, default=0.1,
                      help="KL penalty coefficient for DPO")

  # 生成参数
  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--top_p", type=float, default=0.9)

  # 模型
  parser.add_argument("--model_size", type=str, default='gpt2',
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
  parser.add_argument("--exp_name", type=str, default="dpo_beta0.1")

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-dpo_beta{args.beta}-sonnet.pt'
  seed_everything(args.seed)
  train(args)
