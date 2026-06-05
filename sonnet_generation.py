'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model
from logging_utils import ExperimentLogger

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


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    对一批 sonnet 文本进行前向传播，返回序列中每个 token 的词汇表 logits。

    与 ParaphraseGPT 不同，这里不是只取最后一个 token，而是返回整个序列
    每个位置的 logits。这样模型才能学习到 sonnet 中每个 token 的分布，
    而不仅仅是最后一个 token 之后的预测。

    训练循环会做标准的 next-token prediction：
      - logits[:, :-1]  → 去掉最后一个位置的预测（它没有对应的 label）
      - labels = input_ids[:, 1:]  → 去掉第一个 token，作为 ground truth

    返回:
      (batch_size, seq_len, vocab_size) 的 logits
    """
    # =========================================================================
    # 第 1 步：将输入送入 GPT-2，获取所有 token 的隐藏状态
    # =========================================================================
    # 这次我们使用 'last_hidden_state'（整个序列），而不是 'last_token'（最后一个 token）
    # 因为 sonnet generation 是 next-token prediction 任务，需要每个位置的预测
    
    # 将 tokenized 后的输入送入 GPT-2 模型，首先 embed 得到 embeddings，然后经过 12 层 Transformer
    # 得到所有 token 的 hidden state，最终返回的 last_hidden_state 是所有 token 的上下文感知表示
    outputs = self.gpt(input_ids, attention_mask)
    last_hidden_state = outputs['last_hidden_state']  # (batch_size, seq_len, hidden_size)

    # =========================================================================
    # 第 2 步：将每个 token 的隐藏状态投影到词汇表空间
    # =========================================================================
    # hidden_state_to_token 可以接受任意形状的 hidden state：
    #   (batch_size, seq_len, hidden_size) @ (hidden_size, vocab_size)
    #   = (batch_size, seq_len, vocab_size)
    #
    # 结果中 last_hidden_state[i, j, :] 是第 i 个样本、第 j 个 token 在整个
    # 词汇表上的 logits（未归一化得分）

    # 得到每个位置上每个 token 的未归一化的 logits
    logits = self.gpt.hidden_state_to_token(last_hidden_state)  # (batch_size, seq_len, vocab_size)

    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # 初始化实验日志
  logger = ExperimentLogger(
    experiment_name=args.exp_name,
    task="sonnet",
    method="full_finetune",  # 后续 DPO 时改为 "dpo"
    output_dir="logs"
  )
  logger.log_config(vars(args))
  logger.log_model_info(model)
  logger.log_training_start()

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    logger.log_epoch_start()
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()

    # 生成 dev sonnets 并计算 chrF score
    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
      sonnet_id = batch[0]
      prompt = batch[1]
      encoding = model.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      generated_sonnets.append((sonnet_id, output[1]))
      print(f'{prompt}{output[1]}\n\n')

    # 保存临时文件用于计算 chrF
    temp_dev_path = f'predictions/generated_sonnets_dev_epoch_{epoch}.txt'
    with open(temp_dev_path, 'w') as f:
      f.write('--Generated Sonnets--\n\n')
      for sonnet_id, sonnet_text in generated_sonnets:
        f.write(f'\n{sonnet_id}\n')
        f.write(sonnet_text)

    # 计算 chrF score
    from evaluation import test_sonnet
    chrf_score = test_sonnet(
      test_path=temp_dev_path,
      gold_path='data/TRUE_sonnets_held_out_dev.txt'
    )
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev chrF :: {chrf_score :.3f}")

    # 记录当前 epoch 的指标
    logger.log_epoch_metrics(epoch, {
      "train_loss": train_loss,
      "dev_chrF": chrf_score,
    })

  # 训练结束，记录最终结果
  logger.log_training_end()
  logger.log_final_results({
    "final_chrF": chrf_score,
  })
  logger.save()
  logger.print_summary()

  # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
  save_model(model, optimizer, args, f'{args.epochs}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  parser.add_argument("--exp_name", type=str, default="baseline",
                      help="Experiment name for logging (e.g., 'baseline', 'dpo_beta0.1')")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
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
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)