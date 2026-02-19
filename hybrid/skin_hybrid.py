import numpy as np
import matplotlib.pyplot as plt
from core.projection_model import ProjectionModel

# Annahme: Du hast gewebe_reservoir.py aus dem Original-PilotWeave kopiert
# Falls nicht: Hier ein vereinfachter Placeholder (ersetze durch echten Code aus Repo)
def pilotweave_sim(n_steps=3000, seed=42, plv_threshold=0.7, external_gate=1.0):
    """
    Vereinfachter PilotWeave-Simulator (Placeholder).
    Ersetze durch import aus pilotweave/gewebe_reservoir.py
    """
    rng = np.random.default_rng(seed)
    x, y = np.zeros(n_steps), np.zeros(n_steps)
    x[0], y[0] = 0.0, 0.0
    thrust = 0.0

    for t in range(1, n_steps):
        # Sparse cosmic-ray event (placeholder)
        event = rng.normal(0, 1) if rng.random() < 0.05 else 0.0

        # PLV-Placeholder (normalerweise rolling FFT-Hilbert)
        plv = 0.5 + 0.5 * np.sin(t * 0.01 + event)  # simuliert 0.0–1.0

        # External gate from projection model (z.B. 0 bei cold shower)
        effective_plv = plv * external_gate

        if effective_plv > plv_threshold:
            thrust = 0.05 * effective_plv  # Thrust nur bei hoher Kohärenz
        else:
            thrust = 0.0

        # 2D-Trajektorie (Surfen auf FDM-Wellen)
        angle = rng.uniform(0, 2 * np.pi) * thrust
        x[t] = x[t-1] + np.cos(angle) * thrust
        y[t] = y[t-1] + np.sin(angle) * thrust

    return x, y, thrust  # Rückgabe für Plot + Metrik

def run_hybrid_simulation(doc: float = 0.0, n_steps: int = 3000, seed: int = 42):
    # Projection Model (unsere Bio-Seite)
    model = ProjectionModel(n_steps=n_steps, seed=seed)
    res = model.run(DOC=doc)

    # Hybrid-Kopplung: Cold-Shower + Tc → Gate für PilotWeave
    cold_gate = 1.0 - 0.8 * res["cold_shower"].astype(float)  # 0.2 bei cold shower
    tc_norm = (res["Tc"] - 36.8) / 1.3  # normiert auf Tummo-Range

    # PilotWeave mit externem Gate
    x, y, thrust_final = pilotweave_sim(n_steps=n_steps, seed=seed, external_gate=cold_gate.mean())

    # Plot: Trajektorie + Overlays
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x, y, c=tc_norm, cmap='coolwarm', s=20, alpha=0.7)
    ax.scatter(x[res["cold_shower"]], y[res["cold_shower"]], c='red', s=50, marker='x', label='Cold Shower Events')
    ax.set_title(f"PilotWeave-Skin Hybrid – DOC={doc} | Final Thrust: {thrust_final:.3f}")
    ax.set_xlabel("X (ar
