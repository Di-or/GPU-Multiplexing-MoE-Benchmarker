"""Model components for the MoE expert swapping PoC."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.masked_fill(mask, float("-inf"))
        out = (F.softmax(attn, dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FFN(nn.Module):
    """Feed-forward network. Used both as a standalone dense layer and as each expert."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class Router(nn.Module):
    """Top-1 router: each token is assigned to exactly one expert."""

    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        indices = logits.argmax(dim=-1)
        weights = F.softmax(logits, dim=-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        return indices, weights


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer. Contains N expert FFNs and a router.
    In normal mode, uses its own experts directly.
    When expert_override is provided, uses those instead (for swapping mode).
    """

    def __init__(self, d_model, d_ff, n_experts):
        super().__init__()
        self.router = Router(d_model, n_experts)
        self.experts = nn.ModuleList([FFN(d_model, d_ff) for _ in range(n_experts)])
        self.n_experts = n_experts

    def forward(self, x, expert_override=None):
        indices, weights = self.router(x)
        output = torch.zeros_like(x)
        for eidx in range(self.n_experts):
            mask = (indices == eidx)
            if not mask.any():
                continue
            expert = expert_override[eidx] if expert_override else self.experts[eidx]
            output[mask] = expert(x[mask]) * weights[mask].unsqueeze(-1)
        return output, indices


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, is_moe, n_experts):
        super().__init__()
        self.attn = Attention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.is_moe = is_moe
        self.ffn = MoELayer(d_model, d_ff, n_experts) if is_moe else FFN(d_model, d_ff)

    def forward(self, x, expert_override=None):
        x = x + self.attn(self.ln1(x))
        normed = self.ln2(x)
        if self.is_moe:
            ffn_out, indices = self.ffn(normed, expert_override)
            return x + ffn_out, indices
        return x + self.ffn(normed), None


class ToyMoEModel(nn.Module):
    """
    Decoder-only transformer with MoE layers.

    4 layers total: layers 0,2 are dense FFN, layers 1,3 are MoE.
    Each MoE layer has 4 experts with top-1 routing.
    """

    def __init__(self, vocab_size=1000, d_model=512, n_heads=4, d_ff=2048,
                 n_experts=4, n_layers=4, moe_layers=(1, 3)):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff,
                             is_moe=(i in moe_layers), n_experts=n_experts)
            for i in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.moe_layer_indices = list(moe_layers)

    def forward(self, input_ids):
        """Standard forward pass — all weights on same device."""
        x = self.embed(input_ids)
        all_indices = {}
        for i, block in enumerate(self.blocks):
            x, indices = block.forward(x)
            if indices is not None:
                all_indices[i] = indices
        x = self.ln_f(x)
        return self.head(x), all_indices