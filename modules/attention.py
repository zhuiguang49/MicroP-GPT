import torch

from einops import rearrange
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    '''
    transform 函数对 x 进行线性变换，生成 Query, Key, Value
    '''
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    # 重排维度，将维度转化为 batch, heads, time, dim
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # 首先计算 Query 和 Key 的点积，得到注意力分数(注意要用 scaled dot-product attention)
    attention_scores = torch.matmul(query, key.transpose(-1,-2)) / self.attention_head_size ** 0.5
    # 接着实施掩码，函数已经传入了 attention_mask，直接和 attention_scores 相加即可
    masked_attention_scores = attention_scores + attention_mask
    # 然后对相似度分数应用 softmax 得到 attention 概率，
    attention_probs = F.softmax(masked_attention_scores, dim = -1)
    # 应用 dropout
    dropout_attention_probs = self.dropout(attention_probs)
    # 最后将 dropout 后的 probabilities 和 value 作内积
    attention_value = torch.matmul(dropout_attention_probs, value)
    # 然后将多头的结果合并，得到 [bs, seq_len, hidden_state] 的输出
    attention_value = rearrange(attention_value, 'b h t d -> b t (h d)')
    return attention_value


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
