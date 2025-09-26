# Quantum Sampler Demo

This repository provides a minimal implementation of a **quantum sampler** using **Qiskit Machine Learning** and **PyTorch**.  
The objective is to demonstrate how parameterized quantum circuits can be trained to reproduce target probability distributions, serving as a foundation for **quantum-enhanced diffusion models**.

---

## Overview

- **Motivation**: Classical diffusion models rely on Gaussian noise in the forward process. This work explores replacing classical noise with quantum sampling to leverage superposition and entanglement.  
- **Approach**: A parameterized quantum circuit is wrapped with `SamplerQNN` and connected to PyTorch via `TorchConnector`. Training is performed using gradient-based optimization.  
- **Result**: The quantum sampler successfully converges to a Bell-like distribution, validating the feasibility of quantum sampling as a building block for generative modeling.

---

## Methods

- **Frameworks**:  
  - [Qiskit Machine Learning](https://qiskit.org/documentation/machine-learning/)  
  - [PyTorch](https://pytorch.org/)  

- **Key Components**:  
  - `SamplerQNN`: maps quantum circuits to probability distributions.  
  - `TorchConnector`: integrates QNNs into PyTorch as differentiable modules.  
  - Optimizer: Adam, with KL divergence as the loss function.  

- **Circuit Ansatz**:  
  - `RealAmplitudes(n_qubits=2, reps=1, entanglement='full')`.

---

## Results

- **Target distribution**:  
  \[
  p(00) = 0.5,\; p(11) = 0.5
  \]

- **Final learned distribution** (after 150 epochs):  
  \[
  p(00) = 0.521,\; p(01) = 0.001,\; p(10) = 0.001,\; p(11) = 0.478
  \]

- **KL divergence** decreased from 4.14 → 0.002.  
- Training curve is saved as `training_curve.png`.

---

## Installation

```bash
pip install qiskit qiskit-machine-learning torch matplotlib
