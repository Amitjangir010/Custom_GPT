import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_hid, nlayers, dropout=0.1):
        super().__init__()
        self.model_type = 'GPT'
        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout),
            nlayers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src, src, tgt_mask=src_mask)
        output = self.fc_out(output)
        return output

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.pos_encoder.pe.size(1) else idx[:, -self.pos_encoder.pe.size(1):]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def resize_token_embeddings(self, new_vocab_size):
        old_embed = self.embedding
        new_embed = nn.Embedding(new_vocab_size, self.d_model)
        new_embed.weight.data[:old_embed.weight.shape[0], :] = old_embed.weight.data
        self.embedding = new_embed

        old_fc_out = self.fc_out
        new_fc_out = nn.Linear(self.d_model, new_vocab_size)
        new_fc_out.weight.data[:old_fc_out.weight.shape[0], :] = old_fc_out.weight.data
        new_fc_out.bias.data[:old_fc_out.bias.shape[0]] = old_fc_out.bias.data
        self.fc_out = new_fc_out
