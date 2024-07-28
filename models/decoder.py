import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.mlp = MLP(d_model, d_ff, d_model, dropout=dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # MLP
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm1(x)

        # Self-Attention
        self_attn_output, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm2(x)

        # Cross-Attention
        cross_attn_output, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
