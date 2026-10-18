"""
Spiking Neural Network module for temporal processing.

Modules:

- ``production_snn``: the layered snnTorch network (leaky integrate-and-fire
  layers, an optional spike-timing-dependent plasticity stack, a population
  synchrony readout and guarded persistence); needs the neural extra.
- ``snn_core_optimized``: the vectorized NumPy spiking network that the
  emotion encoder drives by default.
- ``emotion_encoder``: rate coding of an emotional state into spikes for a
  spiking network, and decoding of its output firing rates.
"""
