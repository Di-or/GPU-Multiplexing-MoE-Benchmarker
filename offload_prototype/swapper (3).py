"""
Expert swapper — single GPU slot, synchronous loading.

Simple design:
    - All expert weights live on CPU (source of truth, never modified).
    - One pre-allocated FFN "slot" on GPU holds the active expert.
    - Before running an expert, its weights are copied from CPU to the slot.
    - After computation, the slot is free to be overwritten with the next expert.
"""

import time
import torch
from model import FFN


class ExpertSwapper:
    def __init__(self, model, device):
        self.device = device
        self.stats = {"transfers": 0, "transfer_time": 0.0}

        # Snapshot all expert weights to CPU
        self.cpu_experts = {}
        sample_expert = model.blocks[model.moe_layer_indices[0]].ffn.experts[0]
        d_model = sample_expert.w1.in_features
        d_ff = sample_expert.w1.out_features

        for layer_idx in model.moe_layer_indices:
            for eidx, expert in enumerate(model.blocks[layer_idx].ffn.experts):
                self.cpu_experts[(layer_idx, eidx)] = {
                    k: v.cpu().clone() for k, v in expert.state_dict().items()
                }

        # Single GPU slot — holds one expert at a time
        self.slot = FFN(d_model, d_ff).to(device)

    def load_expert(self, layer_idx, expert_idx):
        """Load an expert into the GPU slot. Returns the FFN module."""
        t0 = time.perf_counter()
        cpu_state = self.cpu_experts[(layer_idx, expert_idx)]
        for name, param in self.slot.named_parameters():
            param.data.copy_(cpu_state[name].to(self.device))
        self.stats["transfer_time"] += time.perf_counter() - t0
        self.stats["transfers"] += 1

        return self.slot