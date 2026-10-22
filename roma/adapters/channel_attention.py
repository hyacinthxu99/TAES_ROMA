import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseAttention(nn.Module):
    def __init__(self, input_size, output_size, seq_length, hidden_size, predict_length=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.predict_length = seq_length if predict_length is None else predict_length

        self.query = nn.Linear(self.seq_length, self.hidden_size)
        self.key = nn.Linear(self.seq_length, self.hidden_size)
        self.value = nn.Linear(self.seq_length, self.seq_length)
        self.out = nn.Linear(self.input_size, self.output_size)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, value), attention

    def forward(self, x, mask=None):
        """Attend over channels of [batch, time, channels] input."""

        x = x.transpose(-2, -1)

        query = self.query(x)
        key = self.key(x)

        value = self.value(x)

        attention_output, attention = self.scaled_dot_product_attention(query, key, value, mask)

        output = self.out(attention_output.transpose(-2, -1))

        return output, attention
