'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
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


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    对一批句子对进行 paraphrase detection（cloze-style 完形填空方式）。

    输入已被数据集格式化为：
      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
    模型需要预测句子末尾的下一个 token 是 "yes"（token id = 8505）
    还是 "no"（token id = 3919）。

    return:
      (batch_size, 2) 的 logits，第一列 = "no"，第二列 = "yes"。
    """
    # =========================================================================
    # 第 1 步：将输入送入 GPT-2，获取每个 token 的隐藏状态
    # =========================================================================
    # self.gpt(...) 返回一个字典，包含：
    #   'last_hidden_state': (batch_size, seq_len, hidden_size)
    #       序列中每个 token 经过 12 层 Transformer 后的上下文表示
    #   'last_token': (batch_size, hidden_size)
    #       最后一个非 padding token 的隐藏状态（models/gpt2.py 中已经实现）
    #
    # 对于 cloze-style 任务，我们只需要最后一个 token 位置的表示，
    # 因为我们要在这个位置预测下一个 token 是 "yes" 还是 "no"
    # self.gpt 实际上调用的是 models/gpt2.py 中的 forward 函数
    outputs = self.gpt(input_ids, attention_mask)
    last_token_hidden = outputs['last_token']  # (batch_size, hidden_size)

    # =========================================================================
    # 第 2 步：将最后一个 token 的隐藏状态投影到词汇表空间，直接利用 models/gpt2.py 
    # 中的 hidden_state_to_token 函数即可
    # =========================================================================
    # hidden_state_to_token 做的事情：
    #   hidden_state @ word_embedding.weight^T
    # 即 (batch_size, hidden_size) @ (hidden_size, vocab_size)
    #  = (batch_size, vocab_size)
    #
    # 结果中每个位置的值代表"该 token 是词汇表中第 i 个单词"的未归一化得分（logit）
    vocab_logits = self.gpt.hidden_state_to_token(last_token_hidden)  # (batch_size, vocab_size)

    # =========================================================================
    # 第 3 步：从整个词汇表（50257 个 token）中只提取 "yes" 和 "no" 的 logits
    # =========================================================================
    # GPT-2 的 BPE tokenizer 中，这两个单词各自是一个完整的 token：
    #   - "yes" → token id = 8505
    #   - "no"  → token id = 3919
    #
    # 我们只关心这两个 token 的得分，因为模型的任务就是判断
    # 接下来应该输出 "yes"（是 paraphrase）还是 "no"（不是 paraphrase）
    yes_logits = vocab_logits[:, 8505].unsqueeze(1)  # (batch_size, 1)
    no_logits = vocab_logits[:, 3919].unsqueeze(1)    # (batch_size, 1)

    # =========================================================================
    # 第 4 步：拼接成二分类 logits 返回
    # =========================================================================
    # 返回形状 (batch_size, 2)：
    #   第 0 列 = "no" 的 logit  → class 0 = not paraphrase
    #   第 1 列 = "yes" 的 logit → class 1 = is paraphrase
    #
    # 这样 torch.argmax(logits, dim=1) 的结果就是 0 或 1，
    # 可以直接和 label 比较计算准确率
    return torch.cat([no_logits, yes_logits], dim=1)  # (batch_size, 2)



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
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # 将 labels 从 token id 转换为 class index
      # 数据集的 collate_fn 将 label 编码为 tokenized 的 "yes"/"no" 字符串，
      # 所以 labels 中的值是 token id：8505 表示 "yes"，3919 表示 "no"
      # 但 F.cross_entropy 要求 label 是 class index（0 或 1），
      # 因此我们将 8505 ("yes") → 1，3919 ("no") → 0
      labels = (labels == 8505).long()  # (batch_size,), 1 = is paraphrase, 0 = not paraphrase

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

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

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

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
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
