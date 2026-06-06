'''
Phase 1: Generate rejected samples for DPO training.

使用未微调的 GPT-2 对训练集 sonnet 的前 3 行做补全，生成低质量版本。
生成的结果保存为 JSON 文件，供 DPO 训练使用。

Running:
  `python generate_rejected_sonnets.py --use_gpu`
'''

import argparse
import json
import random
import torch
import numpy as np

from tqdm import tqdm
from transformers import GPT2Tokenizer

from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from sonnet_generation import SonnetGPT, add_arguments, seed_everything


def get_prompt_from_sonnet(sonnet_text: str, num_prompt_lines: int = 3) -> str:
  """从完整 sonnet 中提取前 num_prompt_lines 行作为 prompt。"""
  lines = [line for line in sonnet_text.strip().split('\n') if line.strip()]
  prompt_lines = lines[:num_prompt_lines]
  return '\n'.join(prompt_lines)


@torch.no_grad()
def generate_rejected(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # 加载未微调的 GPT-2
  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)
  model.eval()

  # 加载训练集 sonnets
  sonnet_dataset = SonnetsDataset(args.sonnet_path)

  paired_data = []
  print(f"Generating rejected samples for {len(sonnet_dataset)} sonnets...")

  for idx in tqdm(range(len(sonnet_dataset))):
    _, sonnet_text = sonnet_dataset[idx]
    prompt = get_prompt_from_sonnet(sonnet_text, num_prompt_lines=3)

    # Tokenize prompt
    encoding = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
    input_ids = encoding['input_ids'].to(device)

    # 用未微调的 GPT-2 生成补全（高 temperature 能够使得模型选择概率较低的输出
    # 增加随机性/低质量）
    _, generated_text = model.generate(
      input_ids,
      temperature=args.temperature,
      top_p=args.top_p,
      max_length=args.max_length
    )

    paired_data.append({
      "id": idx,
      "prompt": prompt,
      "chosen": sonnet_text.strip(),
      "rejected": generated_text.strip(),
    })

  # 保存为 JSON
  with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(paired_data, f, indent=2, ensure_ascii=False)

  print(f"Saved {len(paired_data)} paired samples to {args.output_path}")

  # 打印几个样例
  print("\n--- Sample 0 ---")
  print(f"Prompt:\n{paired_data[0]['prompt']}")
  print(f"\nChosen (real):\n{paired_data[0]['chosen'][:200]}...")
  print(f"\nRejected (generated):\n{paired_data[0]['rejected'][:200]}...")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--output_path", type=str, default="data/sonnets_rejected.json")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--seed", type=int, default=11711)

  # 生成参数：高 temperature 产生更多样化/低质量的文本
  parser.add_argument("--temperature", type=float, default=1.5,
                      help="Higher temperature for more diverse (lower quality) generation")
  parser.add_argument("--top_p", type=float, default=0.95)
  parser.add_argument("--max_length", type=int, default=128,
                      help="Max tokens to generate")

  parser.add_argument("--model_size", type=str, default='gpt2',
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'])

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)
  generate_rejected(args)
