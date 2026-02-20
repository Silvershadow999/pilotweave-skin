import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core.projection_model import ProjectionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Placeholder für PilotWeave (vereinfacht – später echtes gewebe_reservoir importieren)
def pilotweave_sim(
    n_steps: int = 3000,
    seed: int = 42,
    plv_threshold: float = 0.7,
    external_gate: float = 1.0,
    quasi_boost: float = 1.0,
):
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps, dtype=float)
    y = np.zeros(n_steps, dtype=float)
    thrust_history = np.zeros(n_steps, dtype=float)

    for t in range(1, n_steps):
        # Sparse exogenous event (placeholder)
        event = rng.normal(0, 1) if rng.random() < 0.05 else 0.0

        # PLV-Placeholder (simuliert rolling Hilbert)
        plv = 0.5 + 0.5 * np.sin(
            t * 0.01 + event + quasi_boost * np.sin(t * 0.05)
        )

        # External gate (aus ProjectionModel: cold_shower + quasi_boost)
        effective_plv = plv * external_gate * quasi_boost

        thrust = 0.05 * effective_plv if effective_plv > plv_threshold else 0.0

        angle = rng.uniform(0, 2 * np.pi)
        x[t] = x[t - 1] + np.cos(angle) * thrust
        y[t] = y[t - 1] + np.sin(angle) * thrust
        thrust_history[t] = thrust

    return x, y, thrust_history


def run_hybrid_simulation(
    doc: float = 0.0,
    mode: str = "standard",
    n_steps: int = 3000,
    seed: int = 42,
    save_plot: bool = True,
    show_plot: bool = True,
):
    logger.info("Hybrid run | DOC=%s | mode=%s | seed=%s", doc, mode, seed)

    # Projection Model (Bio-Seite)
    model = ProjectionModel(n_steps=n_steps, seed=seed)
    res = model.run(DOC=doc, mode=mode)

    # Erwartete Keys (sanity)
    for k in ("cold_shower", "M", "Tc"):
        if k not in res:
            raise KeyError(f"ProjectionModel.run() result missing key: {k}")

    cold_shower = np.asarray(res["cold_shower"], dtype=bool)
    M = np.asarray(res["M"], dtype=float)
    Tc = np.asarray(res["Tc"], dtype=float)

    # Hybrid Gate: Cold-Shower + Quasicrystal-Boost
    # (cold_shower True -> stronger gating)
    cold_gate = 1.0 - 0.8 * cold_shower.astype(float)  # 0.2 bei cold shower
    external_gate = float(np.mean(cold_gate))

    # quasi_boost: stabiler, falls M viele Nullen hat
    M_pos = M[M > 0]
    if M_pos.size > 0 and np.mean(M_pos) != 0:
        quasi_boost = float(np.mean(M) / np.mean(M_pos))
    else:
        quasi_boost = 1.0

    # PilotWeave mit Gate & Boost
    x, y, thrust_history = pilotweave_sim(
        n_steps=n_steps,
        seed=seed,
        external_gate=external_gate,
        quasi_boost=quasi_boost,
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Trajektorie (Farbkodierung über Tc)
    scatter = ax1.scatter(
        x, y, c=Tc, cmap="coolwarm", s=15, alpha=0.7
    )
    if np.any(cold_shower):
        ax1.scatter(
            x[cold_shower],
            y[cold_shower],
            c="red",
            s=50,
            marker="x",
            label="Cold Shower",
        )

    ax1.set_title(f"PilotWeave-Skin Hybrid | DOC={doc} | Mode={mode}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Tc (°C)")

    # Thrust-History
    ax2.plot(thrust_history, label="Thrust")
    ax2.set_title("Thrust over time (gated by coherence & cold shower)")
    ax2.set_xlabel("Time steps")
    ax2.set_ylabel("Thrust")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_plot:
        out = REPORT_DIR / f"hybrid_doc_{doc}_mode_{mode}.png"
        plt.savefig(out, dpi=220)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    logger.info(
        "Hybrid run done | Avg Tc: %.2f °C | Cold showers: %.1f%% | Max Thrust: %.3f | external_gate=%.3f | quasi_boost=%.3f",
        float(np.mean(Tc)),
        float(np.mean(cold_shower) * 100.0),
        float(np.max(thrust_history)),
        external_gate,
        quasi_boost,
    )


if __name__ == "__main__":
    run_hybrid_simulation(doc=0.0, mode="quasicrystal")
    # run_hybrid_simulation(doc=0.8, mode="quasicrystal")
