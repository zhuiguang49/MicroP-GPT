from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    ### YOUR CODE HERE
    output = dense_layer(output)
    output = dropout(output)
    return output + input

  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """
    # 首先实现 masked_self_attention 层的前向传播，包含三个部分，preNorm，attention, add
    # preNorm
    normed_hidden_states = self.attention_layer_norm(hidden_states)
    # attention
    attention_output = self.self_attention(normed_hidden_states, attention_mask)
    # add
    attention_sublayer_output = self.add(hidden_states, attention_output, self.attention_dense, self.attention_dropout)

    # 接着实现 FFN 子层的前向传播，同样包含三个部分，preNorm, feedForward, add
    # preNorm
    normed_ffn_input = self.out_layer_norm(attention_sublayer_output)
    
    # feedForward
    # interm_dense 是一个全连接层（就是一个 Wx+ b），将维度从 768 维扩展到 3072 维
    ffn_output = self.interm_dense(normed_ffn_input) # 768->3072
    # GELU 激活函数
    ffn_output = self.interm_af(ffn_output)
    # add（add 里面就包括了 outdense，所以不需要在 GELU 之后再添加一次 outdense，否则会造成维度错误）
    ffn_sublayer_output = self.add(attention_sublayer_output, ffn_output, self.out_dense, self.out_dropout)
    return ffn_sublayer_output