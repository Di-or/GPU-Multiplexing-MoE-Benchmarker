"""
Entry point for the MoE expert swapping proof of concept.

Usage:
    python main.py

What it does:
    1. Builds a toy decoder-only MoE transformer (random weights).
    2. Runs a forward pass entirely on CPU as a reference.
    3. Moves shared weights to GPU, sets up expert swapper with one GPU slot.
    4. Runs the same input through the swapping system.
    5. Compares outputs to verify correctness.
    6. Prints transfer statistics.
"""

import time
import torch
from model import ToyMoEModel
from swapper import ExpertSwapper


def forward_with_swapping(model, input_ids, swapper):
    """
    Layer-by-layer forward pass that swaps experts on demand.

    Shared weights (embeddings, attention, layernorm, output head) stay on GPU.
    Expert weights are loaded from CPU one at a time, right before they're needed.
    """
    device = swapper.device
    x = model.embed(input_ids.to(device))

    for i, block in enumerate(model.blocks):
        x = x + block.attn(block.ln1(x))
        normed = block.ln2(x)

        if block.is_moe:
            indices, weights = block.ffn.router(normed)
            needed = indices.unique().tolist()

            output = torch.zeros_like(normed)
            for eidx in needed:
                # Load this expert's weights from CPU to GPU
                expert = swapper.load_expert(i, eidx)

                # Run it on all tokens assigned to this expert
                mask = (indices == eidx)
                if mask.any():
                    output[mask] = expert(normed[mask]) * weights[mask].unsqueeze(-1)

            x = x + output
        else:
            x = x + block.ffn(normed)

    return model.head(model.ln_f(x))


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Build model ---
    model = ToyMoEModel(d_model=512, d_ff=8192, n_experts=4, n_layers=4)
    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(
        p.numel()
        for li in model.moe_layer_indices
        for e in model.blocks[li].ffn.experts
        for p in e.parameters()
    )
    print(f"Total params:  {total_params:,}")
    print(f"Expert params: {expert_params:,} ({expert_params/total_params:.0%})")
    print(f"Shared params: {total_params - expert_params:,}")

    print()

    # --- CPU reference pass ---
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 512))

    with torch.no_grad():
        ref_output, ref_routing = model(input_ids)

    print("Routing decisions (first 10 tokens):")
    for layer_idx, indices in ref_routing.items():
        print(f"  Layer {layer_idx}: {indices.squeeze().tolist()[:10]}...")

    # --- Swapping pass ---
    model.to(device)
    swapper = ExpertSwapper(model, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        swap_output = forward_with_swapping(model, input_ids, swapper)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # --- Verify correctness ---
    max_diff = (ref_output.to(device) - swap_output).abs().max().item()
    print(f"\nMax output difference: {max_diff:.2e}")
    print(f"Outputs match: {max_diff < 1e-5}")

    print(f"\nSwapper stats:")
    print(f"  Transfer time (CPU->GPU): {swapper.stats['transfer_time']*1000:.1f} ms")
    print(f"  Total forward time:       {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    main()