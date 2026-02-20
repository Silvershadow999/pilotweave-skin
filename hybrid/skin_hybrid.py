import numpy as np
import matplotlib.pyplot as plt
import logging
from core.projection_model import ProjectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Placeholder für PilotWeave (vereinfacht – später echtes gewebe_reservoir importieren)
def pilotweave_sim(n_steps=3000, seed=42, plv_threshold=0.7, external_gate=1.0, quasi_boost=1.0):
    rng = np.random.default_rng(seed)
    x, y = np.zeros(n_steps), np.zeros(n_steps)
    thrust_history = np.zeros(n_steps)

    for t in range(1, n_steps):
        # Sparse cosmic-ray event (placeholder)
        event = rng.normal(0, 1) if rng.random() < 0.05 else 0.0

        # PLV-Placeholder (simuliert rolling Hilbert)
        plv = 0.5 + 0.5 * np.sin(t * 0.01 + event + quasi_boost * np.sin(t * 0.05))

        # External gate (aus ProjectionModel: cold_shower + quasi_boost)
        effective_plv = plv * external_gate * quasi_boost

        if effective_plv > plv_threshold:
            thrust = 0.05 * effective_plv
        else:
            thrust = 0.0

        angle = rng.uniform(0, 2 * np.pi) * thrust
        x[t] = x[t-1] + np.cos(angle) * thrust
        y[t] = y[t-1] + np.sin(angle) * thrust
        thrust_history[t] = thrust

    return x, y, thrust_history

def run_hybrid_simulation(doc: float = 0.0, mode: str = "standard", n_steps: int = 3000, seed: int = 42):
    logger.info(f"Hybrid run | DOC={doc} | mode={mode} | seed={seed}")

    # Projection Model (Bio-Seite)
    model = ProjectionModel(n_steps=n_steps, seed=seed)
    res = model.run(DOC=doc, mode=mode)

    # Hybrid Gate: Cold-Shower + Quasicrystal-Boost
    cold_gate = 1.0 - 0.8 * res["cold_shower"].astype(float)  # 0.2 bei cold shower
    quasi_boost = np.mean(res["M"]) / np.mean(res["M"][res["M"] > 0]) if np.any(res["M"] > 0) else 1.0
    external_gate = np.mean(cold_gate)

    # PilotWeave mit Gate & Boost
    x, y, thrust_history = pilotweave_sim(n_steps=n_steps, seed=seed, external_gate=external_gate, quasi_boost=quasi_boost)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Trajektorie
    scatter = ax1.scatter(x, y, c=res["Tc"], cmap='coolwarm', s=15, alpha=0.7)
    ax1.scatter(x[res["cold_shower"]], y[res["cold_shower"]], c='red', s=50, marker='x', label='Cold Shower')
    ax1.set_title(f"PilotWeave-Skin Hybrid | DOC={doc} | Mode={mode}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Tc (°C)")

    # Thrust-History
    ax2.plot(thrust_history, color='purple', label='Thrust')
    ax2.set_title("Thrust over time (gated by coherence & cold shower)")
    ax2.set_xlabel("Time steps")
    ax2.set_ylabel("Thrust")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"reports/hybrid_doc_{doc}_mode_{mode}.png")
    plt.show()

    logger.info(f"Hybrid run done | Avg Tc: {np.mean(res['Tc']):.2f} °C | "
                f"Cold showers: {np.mean(res['cold_shower'])*100:.1f}% | "
                f"Max Thrust: {np.max(thrust_history):.3f}")

if __name__ == "__main__":
    run_hybrid_simulation(doc=0.0, mode="quasicrystal")
    # run_hybrid_simulation(doc=0.8, mode="quasicrystal")
