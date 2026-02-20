#!/usr/bin/env python3
"""
PilotWeave-Skin - Main Entry Point (erweitert für quasicrystal_mode & scenarios)
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

from core.projection_model import ProjectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def check_energy_bookkeeping(tc: np.ndarray, tolerance: float = 1e-6) -> bool:
    """DoD: Energy bookkeeping invariant"""
    if np.any(tc < 0):
        logger.error("Energy invariant violated: Tc < 0 K")
        return False
    return True

def load_scenario(scenario_name: str) -> dict:
    """Lädt ein Szenario aus scenarios/ (zukünftig) – aktuell hardcoded"""
    scenarios = {
        "nominal": {"doc": 0.0, "mode": "standard"},
        "quasicrystal_intakt": {"doc": 0.0, "mode": "quasicrystal"},
        "verkalkt": {"doc": 0.8, "mode": "standard"},
        "quasicrystal_verkalkt": {"doc": 0.8, "mode": "quasicrystal"}
    }
    if scenario_name not in scenarios:
        raise ValueError(f"Scenario '{scenario_name}' nicht gefunden. Verfügbar: {list(scenarios.keys())}")
    return scenarios[scenario_name]

def main():
    parser = argparse.ArgumentParser(description="PilotWeave-Skin Hybrid Simulator")
    parser.add_argument("--doc", type=float, default=None, help="Degree of Calcification (0.0 = intakt)")
    parser.add_argument("--mode", choices=["standard", "quasicrystal"], default="standard", help="Projection mode")
    parser.add_argument("--scenario", type=str, default=None, help="Load preset scenario (overrides --doc/--mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=5000, help="Simulation steps")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    # Scenario override
    if args.scenario:
        scenario = load_scenario(args.scenario)
        doc = scenario["doc"]
        mode = scenario["mode"]
        logger.info(f"Loaded scenario '{args.scenario}': DOC={doc}, mode={mode}")
    else:
        doc = args.doc if args.doc is not None else 0.0
        mode = args.mode

    logger.info(f"Run: DOC={doc} | mode={mode} | seed={args.seed} | steps={args.steps}")

    try:
        model = ProjectionModel(n_steps=args.steps, seed=args.seed)
        res = model.run(DOC=doc, mode=mode)

        # Energy check (DoD)
        if not check_energy_bookkeeping(res["Tc"]):
            logger.error("DoD check failed")
            return 1

        # Simple summary
        summary = {
            "timestamp": timestamp,
            "doc": doc,
            "mode": mode,
            "seed": args.seed,
            "avg_tc": float(np.mean(res["Tc"])),
            "cold_rate": float(np.mean(res["cold_shower"]) * 100),
            "mt_final": float(res["S"][2, -1]),
            "run_status": "success"
        }

        report_path = report_dir / f"{timestamp}_doc{doc}_mode{mode}.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Run successful | Avg Tc: {summary['avg_tc']:.2f} °C | Cold showers: {summary['cold_rate']:.1f}%")
        logger.info(f"Report saved: {report_path}")

    except Exception as e:
        logger.error(f"Run failed: {e}", exc_info=True)
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
