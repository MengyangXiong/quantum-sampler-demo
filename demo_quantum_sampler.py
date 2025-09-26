
# demo_quantum_sampler.py
# Minimal "quantum sampler" demo using Qiskit Machine Learning (SamplerQNN) + PyTorch
# Goal: learn a simple 2-qubit target distribution with a parameterized circuit

import math
import numpy as np
import torch
import torch.nn.functional as F

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

def build_qnn(n_qubits=2, reps=1):
    # Parameterized featureless circuit; weights are the trainable parameters
    qc = RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement="full")
    # NOTE: Sampler does not require explicit measurements for probability outputs
    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=[],                 # no classical input, pure sampling
        weight_params=qc.parameters,     # trainable parameters are circuit weights
        sparse=False,                    # dense probs vector of length 2**n
    )
    return qnn, qc

def train_to_match_distribution(target_probs, epochs=150, lr=0.1, reps=1, seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_qubits = int(math.log2(len(target_probs)))
    qnn, qc = build_qnn(n_qubits=n_qubits, reps=reps)

    torch_qnn = TorchConnector(qnn)  # wrap as torch.nn.Module
    opt = torch.optim.Adam(torch_qnn.parameters(), lr=lr)

    # Prepare target as a batch distribution (batch dimension = 1 here)
    target = torch.tensor(target_probs, dtype=torch.float32).unsqueeze(0)  # shape (1, 2**n)

    # Helper to evaluate current distribution (no gradients)
    @torch.no_grad()
    def current_pred():
        # SamplerQNN has no inputs -> pass empty tensor with shape (batch, 0)
        x = torch.zeros((1, 0), dtype=torch.float32)
        p = torch_qnn(x)  # shape (1, 2**n)
        return p.squeeze(0).detach().cpu().numpy()

    print("Target distribution:", np.round(target.squeeze(0).numpy(), 4))
    print("Initial pred      :", np.round(current_pred(), 4))

    # Use KL divergence between model distribution and target distribution
    eps = 1e-8
    history = []
    for epoch in range(1, epochs + 1):
        x = torch.zeros((1, 0), dtype=torch.float32)
        pred = torch_qnn(x)  # probabilities (sum to 1)
        # Clamp to avoid log(0)
        pred = pred.clamp(min=eps, max=1.0)
        # batchmean KL(target || pred): need log(pred)
        loss = F.kl_div(pred.log(), target, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            with torch.no_grad():
                kl = F.kl_div(pred.log(), target, reduction="batchmean").item()
            history.append((epoch, kl))
            print(f"[{epoch:4d}] KL={kl:.6f}  pred={np.round(pred.squeeze(0).detach().cpu().numpy(), 4)}")

    final_pred = current_pred()
    print("Final pred        :", np.round(final_pred, 4))
    return qc, torch_qnn, np.array(history), final_pred

def pretty_bitstrings(n):
    return [format(i, f"0{n}b") for i in range(2**n)]

if __name__ == "__main__":
    # ======= Config =======
    # Example 2-qubit "Bell-like" target: p(00)=0.5, p(11)=0.5
    target_probs = np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float32)
    epochs = 150
    lr = 0.1
    reps = 1
    seed = 7

    qc, model, history, final_pred = train_to_match_distribution(
        target_probs=target_probs, epochs=epochs, lr=lr, reps=reps, seed=seed
    )

    # Print a tiny meeting-ready summary
    n_qubits = int(math.log2(len(target_probs)))
    bit_labels = pretty_bitstrings(n_qubits)
    print("\n=== MEETING SUMMARY ===")
    print("Parameterized ansatz  :", f"RealAmplitudes(n_qubits={n_qubits}, reps={reps}, entanglement='full')")
    print("Optimizer             :", "Adam(lr=%.3g)" % lr)
    print("Epochs                :", epochs)
    print("Target (bitstring: p) :", ", ".join(f"{b}:{p:.3f}" for b, p in zip(bit_labels, target_probs)))
    print("Learned (bit:p)       :", ", ".join(f"{b}:{p:.3f}" for b, p in zip(bit_labels, final_pred)))

    # Optional: plot (only if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        steps, kls = history[:,0], history[:,1]
        plt.figure()
        plt.plot(steps, kls)
        plt.title("KL divergence (target || model)")
        plt.xlabel("Epoch")
        plt.ylabel("KL")
        plt.tight_layout()
        plt.savefig("training_curve.png")
        print("\nSaved training curve to training_curve.png")
    except Exception as e:
        print("Plot skipped:", e)
