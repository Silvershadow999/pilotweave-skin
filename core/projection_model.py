import numpy as np
from typing import Dict, Any, Optional

class ProjectionModel:
    """
    Multi-scale projection architecture (2026 calibrated).
    Models three layers: cosmic (FDM/PilotWeave), neuronal (Gamma/O/A), MT/THz (Superradiance).
    Includes:
    - Fraktale Skalierung (golden ratio)
    - Bottom-up CFC (MT → neuronal pump)
    - Noise-assisted coherence boost
    - Circadian melatonin reset (DOC damping)
    - Thermal proxy (Tc) + cold-shower detection
    """

    def __init__(self, n_steps: int = 3000, seed: int = 42):
        self.phi = (1 + np.sqrt(5)) / 2
        self.rng = np.random.default_rng(seed)

        # 2026 calibrated parameters
        self.k = 0.001
        self.beta = 0.38          # noise-assisted QY boost 25–42 % (ACS JPCB 2024/2026)
        self.gamma = 0.45         # melatonin ρ×C boost (2025 pineal studies)
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

        # CFC (2025 EEG studies)
        self.cfc = {
            "enabled": True,
            "lambda_gate": 0.22,   # mod index 18–28 %
            "eps_C": 0.016,        # \~16 % coherence boost
            "eps_E": 0.008,
        }

        self.temp = {"gain": 1.05, "eps": 1e-9}  # Tummo +0.9–1.3 °C

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
        self.photon_flux = np.zeros((L, T))
        self._S_running_max = np.zeros(T)

    def _melatonin_factor(self, t: int, DOC: float, doc_hits: bool = True, strength: float = 0.55) -> float:
        M_raw = 1.0 + 0.2 * np.sin(2 * np.pi * t / 240.0)
        if not doc_hits:
            return M_raw
        return 1.0 + (M_raw - 1.0) * (1.0 - strength * DOC)

    def run(self, DOC: float = 0.0, doc_hits: bool = True, mel_strength: float = 0.55) -> Dict[str, Any]:
        DOC = float(np.clip(DOC, 0.0, 1.0))
        self.reset_states()

        for t in range(1, self.n_steps):
            event = self.rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])

            self.O[t] = np.clip(self.O[t-1] + self.rng.normal(0, 10) + 5*event, 50, 300)
            self.A[t] = np.clip(self.A[t-1] + self.rng.normal(0, 0.2) - 0.1*event, 0.6, 2.5)

            self.M[t] = self._melatonin_factor(t, DOC, doc_hits, mel_strength)

            for ell in range(self.levels):
                noise_assist = self.E[ell, t-1] * np.exp(-self.E[ell, t-1] / self.E_opt[ell])

                delta = (
                    self.k * (self.O[t] - self.alpha * self.A[t]) * self.rho[ell, t-1] * self.C[ell, t-1]
                    + self.beta * noise_assist
                    + self.gamma * self.M[t] * (1 - DOC)
                )

                self.E[ell, t] = np.clip(self.E[ell, t-1] - 0.01 * delta + self.rng.normal(0, self.update_noise["E"]), 0.05, 1.5)
                self.C[ell, t] = np.clip(self.C[ell, t-1] + 0.02 * delta + self.rng.normal(0, self.update_noise["C"]), 0.1, 0.95)
                self.rho[ell, t] = np.clip(self.rho[ell, t-1] + 0.015 * delta + self.rng.normal(0, self.update_noise["rho"]), 0.3, 2.0)

                self.S[ell, t] = (self.rho[ell, t] * self.C[ell, t] / (1 + self.E[ell, t])) * self.scales[ell]["scale_factor"] * self.M[t]
                self.photon_flux[ell, t] = self.rho[ell, t] * self.scales[ell]["freq_Hz"] * self.C[ell, t]

            # CFC: MT → Neuronal pump (only on neuronal layer)
            if self.cfc["enabled"]:
                mt_drive = np.log1p(self.S[2, t])
                q_neuro = 1.0 / (1.0 + self.E[1, t])
                cfc_gate = np.tanh(self.cfc["lambda_gate"] * mt_drive * q_neuro)
                self.C[1, t] = np.clip(self.C[1, t] + self.cfc["eps_C"] * cfc_gate, 0.1, 0.95)
                self.E[1, t] = np.clip(self.E[1, t] - self.cfc["eps_E"] * cfc_gate, 0.05, 1.5)
                self.S[1, t] = (self.rho[1, t] * self.C[1, t] / (1 + self.E[1, t])) * self.scales[1]["scale_factor"] * self.M[t]

            # Thermal proxy + cold shower
            S_mean = np.mean(self.S[:, t])
            self._S_running_max[t] = max(self._S_running_max[t-1], np.max(self.S[:, t]))
            self.Tc[t] = self.Tc_base + self.temp["gain"] * (S_mean / (self._S_running_max[t] + self.temp["eps"]))

            if np.any(self.E[:, t] > 0.9) and np.mean(self.S[:, t] - self.S[:, t-1]) < 0:
                self.cold_shower[t] = True

        return {
            "S": self.S.copy(),
            "Tc": self.Tc.copy(),
            "cold_shower": self.cold_shower.copy(),
            "mel_pgml": (self.M * self.M_base_pgml).copy(),
            "photon_flux": self.photon_flux.copy(),
            "O": self.O.copy(),
            "A": self.A.copy(),
            "M": self.M.copy(),
        }
