'''
Phase 1: Generate rejected samples for DPO training (Hard Negatives Version).

采用“破坏正样本”策略生成高质量负样本。
通过对真实十四行诗进行打乱顺序、同义词替换或截断，生成“看似合理但存在瑕疵”的文本。
生成的结果保存为 JSON 文件，供 DPO 训练使用。

Running:
  `python generate_rejected_sonnets.py`
'''

import argparse
import json
import random
import re

from tqdm import tqdm
from datasets import SonnetsDataset


def get_prompt_from_sonnet(sonnet_text: str, num_prompt_lines: int = 3) -> str:
  """从完整 sonnet 中提取前 num_prompt_lines 行作为 prompt。"""
  lines = [line for line in sonnet_text.strip().split('\n') if line.strip()]
  prompt_lines = lines[:num_prompt_lines]
  return '\n'.join(prompt_lines)


def perturb_sonnet(sonnet_text: str) -> str:
  """
  对十四行诗进行扰动，生成一个质量较低的版本。
  策略：
  1. 打乱最后两行或最后四行的顺序。
  2. 简单词汇替换（降低艺术性）。
  """
  lines = [line for line in sonnet_text.strip().split('\n') if line.strip()]
  
  if len(lines) < 14:
    return sonnet_text

  # 策略选择：50% 概率打乱顺序，50% 概率替换词汇
  strategy = random.choice(['shuffle', 'replace'])

  if strategy == 'shuffle':
    # 保持前 10 行不变，打乱后 4 行
    prefix = lines[:10]
    suffix = lines[10:]
    random.shuffle(suffix)
    perturbed_lines = prefix + suffix
  else:
    # 简单同义词替换映射
    replacements = {
      r'\blove\b': 'like',
      r'\bfair\b': 'good',
      r'\bbeauty\b': 'look',
      r'\bheart\b': 'mind',
      r'\bsweet\b': 'nice',
      r'\bthou\b': 'you',
      r'\bthy\b': 'your',
      r'\bthee\b': 'you',
      r'\bdoth\b': 'does',
      r'\bhath\b': 'has'
    }
    
    perturbed_lines = []
    for line in lines:
      new_line = line
      for pattern, repl in replacements.items():
        new_line = re.sub(pattern, repl, new_line, flags=re.IGNORECASE)
      perturbed_lines.append(new_line)

  return '\n'.join(perturbed_lines)


def generate_rejected(args):
  # 加载训练集 sonnets
  sonnet_dataset = SonnetsDataset(args.sonnet_path)

  paired_data = []
  print(f"Generating rejected samples for {len(sonnet_dataset)} sonnets...")

  for idx in tqdm(range(len(sonnet_dataset))):
    _, sonnet_text = sonnet_dataset[idx]
    prompt = get_prompt_from_sonnet(sonnet_text, num_prompt_lines=3)
    
    # 生成扰动后的负样本
    rejected_text = perturb_sonnet(sonnet_text)

    paired_data.append({
      "id": idx,
      "prompt": prompt,
      "chosen": sonnet_text.strip(),
      "rejected": rejected_text.strip(),
    })

  # 保存为 JSON
  with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(paired_data, f, indent=2, ensure_ascii=False)

  print(f"✅ Saved {len(paired_data)} paired samples to {args.output_path}")

  # 打印几个样例检查质量
  print("\n--- Sample 0 ---")
  print(f"Prompt:\n{paired_data[0]['prompt']}")
  print(f"\nChosen (real):\n{paired_data[0]['chosen'][:200]}...")
  print(f"\nRejected (perturbed Hard Negative):\n{paired_data[0]['rejected'][:200]}...")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--output_path", type=str, default="data/sonnets_rejected.json")
  parser.add_argument("--seed", type=int, default=11711)

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = get_args()
  random.seed(args.seed)
  generate_rejected(args)
