import numpy as np
from typing import Dict, Any

class ProjectionModel:
    """
    Multi-scale projection architecture (2026 calibrated)
    NEU: quasicrystal_mode mit multi-frequency temporal order (NV-Centers 2025)
    """

    def __init__(self, n_steps: int = 3000, seed: int = 42):
        self.phi = (1 + np.sqrt(5)) / 2
        self.rng = np.random.default_rng(seed)

        self.k = 0.001
        self.beta = 0.38
        self.gamma = 0.45
        self.alpha = 1.0

        self.levels = 3
        self.n_steps = n_steps

        self.scales = [
            {"name": "Kosmisch", "freq_Hz": 1e-3,  "scale_factor": 1.0,   "level": 0},
            {"name": "Neuronal", "freq_Hz": 40.0,  "scale_factor": 4e4,   "level": 1},
            {"name": "MT/THz",   "freq_Hz": 1e12,  "scale_factor": 1e15,  "level": 2},
        ]

        self.E_opt = np.array([0.6, 0.4, 0.25])

        self.Tc_base = 36.8
        self.M_base_pgml = 50.0

        self.cfc = {"enabled": True, "lambda_gate": 0.22, "eps_C": 0.016, "eps_E": 0.008}
        self.temp = {"gain": 1.05, "eps": 1e-9}
        self.update_noise = {"E": 0.05, "C": 0.03, "rho": 0.05}

        self.reset_states()

    def reset_states(self):
        L, T = self.levels, self.n_steps
        self.rho = np.ones((L, T)) * 1.0
        self.C   = np.ones((L, T)) * 0.6
        self.E   = np.ones((L, T)) * 0.4
        self.O   = np.ones(T) * 80.0
        self.A   = np.ones(T) * 1.0
        self.M   = np.ones(T) * 1.0
        self.S   = np.zeros((L, T))
        self.Tc  = np.full(T, self.Tc_base)
        self.cold_shower = np.zeros(T, dtype=bool)

    def run(self, DOC: float = 0.0, mode: str = "standard"):
        """mode = 'standard' oder 'quasicrystal'"""
        self.reset_states()

        for t in range(1, self.n_steps):
            event = self.rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])

            self.O[t] = np.clip(self.O[t-1] + self.rng.normal(0, 10) + 5*event, 50, 300)
            self.A[t] = np.clip(self.A[t-1] + self.rng.normal(0, 0.2) - 0.1*event, 0.6, 2.5)

            # === NEU: Time-Quasicrystal Mode ===
            if mode == "quasicrystal":
                # Multi-frequency quasiperiodic driver (NV-Center Style)
                freqs = np.array([0.618, 1.0, 1.618]) * 2 * np.pi
                phases = self.rng.uniform(0, 2*np.pi, 3)
                drive = np.sum([np.sin(f * t + p) for f, p in zip(freqs, phases)])
                M_raw = 1.0 + 0.15 * drive / 3.0
            else:
                M_raw = 1.0 + 0.2 * np.sin(2 * np.pi * t / 240)

            self.M[t] = M_raw * (1 - DOC)

            for ell in range(self.levels):
                noise_assist = self.E[ell, t-1] * np.exp(-self.E[ell, t-1] / self.E_opt[ell])

                delta = (self.k * (self.O[t] - self.alpha * self.A[t]) * self.rho[ell, t-1] * self.C[ell, t-1] +
                         self.beta * noise_assist +
                         self.gamma * self.M[t])

                self.E[ell, t] = np.clip(self.E[ell, t-1] - 0.01 * delta + self.rng.normal(0, self.update_noise["E"]), 0.05, 1.5)
                self.C[ell, t] = np.clip(self.C[ell, t-1] + 0.02 * delta + self.rng.normal(0, self.update_noise["C"]), 0.1, 0.95)
                self.rho[ell, t] = np.clip(self.rho[ell, t-1] + 0.015 * delta + self.rng.normal(0, self.update_noise["rho"]), 0.3, 2.0)

                scale_factor = self.phi ** ell
                self.S[ell, t] = (self.rho[ell, t] * self.C[ell, t] / (1 + self.E[ell, t])) * scale_factor * self.M[t]

            # Thermal + Cold Shower
            S_mean = np.mean(self.S[:, t])
            self.Tc[t] = self.Tc_base + self.temp["gain"] * (S_mean / (np.max(self.S) + 1e-9))

            if np.any(self.E[:, t] > 0.9) and np.mean(self.S[:, t] - self.S[:, t-1]) < 0:
                self.cold_shower[t] = True

        return {
            "S": self.S.copy(),
            "Tc": self.Tc.copy(),
            "cold_shower": self.cold_shower.copy(),
            "M": self.M.copy()
        }
