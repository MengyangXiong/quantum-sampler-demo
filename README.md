
# Quickstart: Minimal Quantum Sampler (Qiskit Machine Learning + PyTorch)

## 1) Install
```bash
pip install qiskit qiskit-machine-learning torch
```

## 2) Run the demo
```bash
python demo_quantum_sampler.py
```

## 3) What it does
- Builds a 2-qubit parameterized circuit (`RealAmplitudes`).
- Wraps it with `SamplerQNN` and `TorchConnector`, so it behaves like a PyTorch module.
- Trains the circuit parameters to match a target distribution (default: 50% `00`, 50% `11`) with KL divergence.
- Prints distributions before/after training. Saves a `training_curve.png` if matplotlib is available.

## 4) Customize quickly
- Change `target_probs` near the bottom of `demo_quantum_sampler.py` to any length-`2**n` distribution.
- Increase `reps` for a more expressive circuit, or tweak `epochs`/`lr`.
- For more qubits: set a target of length `2**n` and the script infers `n` automatically.

## 5) Talking points for your meeting
- **API choice**: `SamplerQNN` (+ `TorchConnector`) from `qiskit-machine-learning` gives an out-of-the-box, differentiable quantum sampler.
- **Why quantum?**: it replaces the classical noise/ sampler step with a trainable quantum process; easy to plug into a diffusion-like pipeline later.
- **Where to go next**: add dissipative channels/Lindbladian steps and plug your reverse process (Petz map) on top of this training loop.
