'''
Phase 1: Generate rejected samples for DPO training (SFT Model Version).

使用 SFT 微调过的模型，配合略高的 temperature，生成“看似合理但存在瑕疵”的十四行诗，
作为 DPO 训练的高质量负样本 (Hard Negatives)。

Running:
  `python generate_rejected_sonnets_sft.py --use_gpu --sft_checkpoint "best_10-1e-05-sonnet.pt"`
'''

import argparse
import json
import os
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

    args = add_arguments(args)
    model = SonnetGPT(args)
    
    # =========================================================================
    # 核心步骤：加载 SFT 模型权重
    # =========================================================================
    if args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
        print(f"Loading SFT checkpoint from {args.sft_checkpoint} for Hard Negatives...")
        saved = torch.load(args.sft_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved['model'])
    else:
        print("⚠️ WARNING: No valid SFT checkpoint found! Using base GPT-2.")
        print("This will likely produce gibberish instead of hard negatives.")
        
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

        # 使用 SFT 模型生成补全
        token_ids, _ = model.generate(
            input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length
        )
        
        # 解码生成的文本
        generated_text = model.tokenizer.decode(token_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)

        paired_data.append({
            "id": idx,
            "prompt": prompt,
            "chosen": sonnet_text.strip(),
            "rejected": generated_text.strip(),
        })

    # 保存为 JSON
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(paired_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(paired_data)} paired samples to {args.output_path}")

    # 打印几个样例检查质量
    print("\n--- Sample 0 ---")
    print(f"Prompt:\n{paired_data[0]['prompt']}")
    print(f"\nChosen (real):\n{paired_data[0]['chosen'][:200]}...")
    print(f"\nRejected (generated Hard Negative):\n{paired_data[0]['rejected'][:200]}...")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--output_path", type=str, default="data/sonnets_rejected_sft.json")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--seed", type=int, default=11711)
    
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="Path to the SFT model (e.g., best_10-1e-05-sonnet.pt)")

    # 生成参数：将 temperature 从 1.5 降到 1.2 或 1.3
    # SFT 模型在 1.2-1.3 的温度下，会尝试保持诗歌格式，但会犯逻辑和押韵的错误
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Slightly higher temperature for SFT model to make mistakes")
    parser.add_argument("--top_p", type=float, default=0.9)
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
