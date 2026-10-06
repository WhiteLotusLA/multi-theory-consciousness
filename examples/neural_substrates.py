#!/usr/bin/env python3
"""
Neural Substrate Pipeline
=========================

Runs the three neural substrates the framework builds on, each scaled down so
the example finishes in seconds on a CPU:

1. SNN: ``ProductionSNN`` with a small configuration (the default production
   network has 5,196 leaky integrate-and-fire neurons in layers of 1000,
   2048, 2048 and 100).
2. LSM: the compact ``LiquidStateMachine`` with a 200-unit reservoir (the
   production machine, ``ProductionLSM``, has 5,000).
3. HTM: the compact ``HTM`` with 512 columns of 16 cells (``ProductionHTM``
   has 4,096 columns of 32 cells).

The numbers printed are what one short run produced; a spike rate of zero or
an anomaly score of one on random input is a reading, not a fault. In the
runtime the neural orchestrator runs the production substrates and keeps
their latest signals, which each workspace cycle receives.

Needs the ``neural`` extra (torch, snntorch, reservoirpy). Runs on the CPU and
opens no network connection.

Usage:
    MTC_DEVICE=cpu python examples/neural_substrates.py
"""

import logging
import time

import numpy as np
import torch

from mtc.neural.htm.htm_core import HTM
from mtc.neural.liquid.lsm_core import LiquidStateMachine
from mtc.neural.spiking.production_snn import ProductionSNN, SNNConfig

logging.basicConfig(level=logging.WARNING)


def main():
    rng = np.random.default_rng(7)
    torch.manual_seed(7)

    print("\n" + "=" * 60)
    print("  NEURAL SUBSTRATE PIPELINE")
    print("=" * 60)

    # 1. Spiking neural network
    print("\n  [SNN] Spiking neural network (scaled down)")
    snn_config = SNNConfig(
        input_dim=100,
        hidden_layers=[200],
        output_dim=50,
        num_neurons=500,
        timesteps=25,
        use_gpu=False,
    )
    snn = ProductionSNN(config=snn_config)
    snn_input = torch.randn(1, snn_config.input_dim)
    t0 = time.perf_counter()
    snn_output, snn_state = snn.process_input(snn_input)
    snn_ms = (time.perf_counter() - t0) * 1000
    print(f"    Output shape:     {tuple(snn_output.shape)}")
    print(f"    Spike rate:       {snn_state.get('output_spike_rate', 0):.3f}")
    print(f"    Processing time:  {snn_ms:.1f} ms (this run)")

    # 2. Liquid state machine
    print("\n  [LSM] Liquid state machine (scaled down)")
    lsm = LiquidStateMachine(
        n_input=10,
        n_reservoir=200,
        n_output=10,
        spectral_radius=0.9,
    )
    sequence = [rng.standard_normal(10) * 0.5 for _ in range(20)]
    t0 = time.perf_counter()
    for signal in sequence:
        reservoir_state, readout = lsm.step(signal)
    lsm_ms = (time.perf_counter() - t0) * 1000
    print(f"    Reservoir size:   {lsm.n_reservoir}")
    print(
        f"    Reservoir state:  mean={reservoir_state.mean():.4f}, "
        f"std={reservoir_state.std():.4f}"
    )
    print(f"    Readout shape:    {readout.shape}")
    print(f"    Sequence length:  {len(sequence)} steps")
    print(f"    Processing time:  {lsm_ms:.1f} ms (this run)")

    # 3. Hierarchical temporal memory
    print("\n  [HTM] Hierarchical temporal memory (scaled down)")
    htm = HTM(
        input_size=100,
        n_columns=512,
        cells_per_column=16,
        sparsity=0.02,
    )
    patterns = [rng.random(100) for _ in range(10)]
    t0 = time.perf_counter()
    results = [htm.compute(pattern, learn=True) for pattern in patterns]
    htm_ms = (time.perf_counter() - t0) * 1000
    last = results[-1]
    print(f"    Columns:          {htm.spatial_pooler.n_columns}")
    print(f"    SDR sparsity:     {last['sdr'].sparsity:.3f}")
    print(f"    Active cells:     {len(last['active_cells'])}")
    print(f"    Anomaly score:    {last['anomaly_score']:.3f}")
    print(f"    Patterns seen:    {len(patterns)}")
    print(f"    Processing time:  {htm_ms:.1f} ms (this run)")

    print("\n  Each substrate processed its input once.")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
