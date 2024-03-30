import torch
import torch.nn as nn
import einops as ein
from zeta import MambaBlock


class VisionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [MambaBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.projection = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.projection(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class Projector(nn.Module):
    def __init__(self, vision_size, text_size, hidden_size):
        super().__init__()
        self.vision_projection = nn.Linear(vision_size, hidden_size)
        self.text_projection = nn.Linear(text_size, hidden_size)

    def forward(self, vision_features, text_features):
        vision_features = self.vision_projection(vision_features)
        text_features = self.text_projection(text_features)
        return vision_features, text_features


class CobraMLLM(nn.Module):
    def __init__(
        self, vision_size, text_size, hidden_size, num_layers, num_heads, dropout, **_
    ):
        super().__init__()
        self.vision_encoder = VisionEncoder(
            vision_size, hidden_size, num_layers, num_heads, dropout
        )
        self.text_encoder = nn.Embedding(text_size, hidden_size)
        self.projector = Projector(hidden_size, hidden_size, hidden_size)

    def forward(self, vision_input, text_input):
        vision_features = self.vision_encoder(vision_input)
        text_features = self.text_encoder(text_input)
        vision_features, text_features = self.projector(vision_features, text_features)

        # Einsum operation for efficient matrix multiplication
        attn_weights = torch.einsum("bld,bvd->blv", text_features, vision_features)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Einops operation for efficient tensor manipulation
        attended_vision = ein.einsum("blv,bvd->bld", attn_weights, vision_features)
        output = attended_vision + text_features
        return output
