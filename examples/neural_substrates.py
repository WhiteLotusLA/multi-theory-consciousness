#!/usr/bin/env python3
"""
Neural Substrate Pipeline
==========================

Demonstrates the three neural substrate layers that underpin the
consciousness framework:

  1. SNN  — Spiking Neural Network (5,000 LIF neurons, STDP learning)
  2. LSM  — Liquid State Machine (1,000-neuron reservoir, subconscious)
  3. HTM  — Hierarchical Temporal Memory (spatial pooling + sequence learning)

Each substrate processes input differently.  Their outputs can be
fed into the Global Workspace as candidates competing for conscious
access.

Requirements:
    pip install torch snntorch numpy

Usage:
    python examples/neural_substrates.py
"""

import logging
import time
import numpy as np
import torch

from mtc.neural.spiking.production_snn import ProductionSNN, SNNConfig
from mtc.neural.liquid.lsm_core import LiquidStateMachine
from mtc.neural.htm.htm_core import HTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 60)
    print("  NEURAL SUBSTRATE PIPELINE")
    print("=" * 60)

    # ── 1. Spiking Neural Network ────────────────────────────────────
    print("\n  [SNN] Initializing Spiking Neural Network ...")
    snn_config = SNNConfig(
        input_dim=100,          # Smaller for demo
        hidden_layers=[200],    # Single hidden layer
        output_dim=50,
        num_neurons=500,        # Scaled down for quick demo
        timesteps=25,
        use_gpu=False,          # CPU for portability
    )
    snn = ProductionSNN(config=snn_config)

    snn_input = torch.randn(1, snn_config.input_dim)  # batch_size=1
    t0 = time.time()
    snn_output, snn_state = snn.process_input(snn_input)
    snn_ms = (time.time() - t0) * 1000

    print(f"    Output shape:     {snn_output.shape}")
    print(f"    Spike rate:       {snn_state.get('output_spike_rate', 0):.3f}")
    print(f"    Processing time:  {snn_ms:.1f} ms")

    # ── 2. Liquid State Machine ──────────────────────────────────────
    print("\n  [LSM] Initializing Liquid State Machine ...")
    lsm = LiquidStateMachine(
        n_input=10,
        n_reservoir=200,   # Scaled down for demo
        n_output=10,
        spectral_radius=0.9,
    )

    # Feed a short sequence through the reservoir
    lsm_input_sequence = [np.random.randn(10) * 0.5 for _ in range(20)]
    t0 = time.time()
    for signal in lsm_input_sequence:
        reservoir_state, readout = lsm.step(signal)
    lsm_ms = (time.time() - t0) * 1000

    print(f"    Reservoir size:   {lsm.n_reservoir}")
    print(f"    Reservoir state:  mean={reservoir_state.mean():.4f}, "
          f"std={reservoir_state.std():.4f}")
    print(f"    Readout shape:    {readout.shape}")
    print(f"    Sequence length:  {len(lsm_input_sequence)} steps")
    print(f"    Processing time:  {lsm_ms:.1f} ms")

    # ── 3. Hierarchical Temporal Memory ──────────────────────────────
    print("\n  [HTM] Initializing Hierarchical Temporal Memory ...")
    htm = HTM(
        input_size=100,
        n_columns=512,       # Scaled down for demo
        cells_per_column=16,
        sparsity=0.02,
    )

    # Process a few patterns to let HTM learn sequences
    t0 = time.time()
    patterns = [np.random.rand(100) for _ in range(10)]
    results = []
    for pattern in patterns:
        result = htm.compute(pattern, learn=True)
        results.append(result)
    htm_ms = (time.time() - t0) * 1000

    last = results[-1]
    print(f"    Columns:          {htm.spatial_pooler.n_columns}")
    print(f"    SDR sparsity:     {last['sdr'].sparsity:.3f}")
    print(f"    Active cells:     {len(last['active_cells'])}")
    print(f"    Anomaly score:    {last['anomaly_score']:.3f}")
    print(f"    Patterns learned: {len(patterns)}")
    print(f"    Processing time:  {htm_ms:.1f} ms")

    # ── 4. Summary ───────────────────────────────────────────────────
    print("\n  Summary:")
    print("    All three substrates processed input successfully.")
    print("    In a full system, their outputs become WorkspaceCandidates")
    print("    that compete for conscious access in the Global Workspace.")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
