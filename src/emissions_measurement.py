"""
emissions_measurement.py
========================
Measures the actual carbon footprint (kgCO2e) of representative AI workloads
using the CodeCarbon library.

Three workload tiers are provided, matching typical use cases in the dataset:

  Tier 1 — Lightweight inference (customer support bot, translation assistant)
            Small matrix ops, ~100ms per query
  Tier 2 — Medium inference (code assistant, fraud detection)
            Larger matrix ops representing a mid-size transformer inference pass
  Tier 3 — Heavy training/large inference (climate forecasting, medical imaging)
            Sustained compute load over several seconds

Each tier is run for a fixed number of iterations.  The measured kgCO2e is
printed and written to data/emissions_measured.csv for comparison with the
estimated values in usecases.csv.

Usage:
    python src/emissions_measurement.py

Requirements:
    pip install codecarbon
"""

import csv
import os
import time

import numpy as np

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("⚠  codecarbon not installed. Install with: pip install codecarbon")
    print("   Running workloads without measurement for timing reference only.\n")


# ── Workload functions ────────────────────────────────────────────────────────

def workload_tier1(n_iters: int = 500):
    """
    Tier 1: Lightweight inference simulation.
    Represents a small transformer (e.g., 7B param) doing single-sentence
    classification or short text generation.
    Matrix size ~512x512 — similar compute to running one forward pass of a
    small language model on a CPU.
    """
    for _ in range(n_iters):
        a = np.random.randn(512, 512).astype(np.float32)
        b = np.random.randn(512, 512).astype(np.float32)
        _ = a @ b
        # Simulate attention mechanism: softmax over rows
        _ = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)


def workload_tier2(n_iters: int = 200):
    """
    Tier 2: Medium inference simulation.
    Represents a medium-size model (e.g., BERT-large equivalent) processing
    a batch of inputs.  Matrix size ~2048x2048.
    """
    for _ in range(n_iters):
        a = np.random.randn(2048, 2048).astype(np.float32)
        b = np.random.randn(2048, 2048).astype(np.float32)
        _ = a @ b
        # Simulate layer normalisation
        mean = a.mean(axis=-1, keepdims=True)
        std  = a.std(axis=-1, keepdims=True) + 1e-6
        _ = (a - mean) / std


def workload_tier3(duration_seconds: float = 5.0):
    """
    Tier 3: Heavy / sustained compute simulation.
    Represents a large scientific model (climate forecasting, medical imaging)
    running extended inference or a short training step.
    Runs for a fixed wall-clock duration rather than a fixed number of
    iterations, to better simulate sustained compute.
    """
    start = time.perf_counter()
    while time.perf_counter() - start < duration_seconds:
        a = np.random.randn(4096, 4096).astype(np.float32)
        b = np.random.randn(4096, 4096).astype(np.float32)
        _ = a @ b


# ── Measurement harness ───────────────────────────────────────────────────────

def measure_workload(name: str, fn, description: str) -> dict:
    """
    Run a workload function under CodeCarbon tracking (if available) and
    return a results dict.
    """
    print(f"\n{'─' * 55}")
    print(f"  Measuring: {name}")
    print(f"  Workload:  {description}")
    print(f"{'─' * 55}")

    t_start = time.perf_counter()

    if CODECARBON_AVAILABLE:
        # log_level='error' suppresses CodeCarbon's verbose output
        tracker = EmissionsTracker(
            project_name=f"ai_misuse_{name.lower().replace(' ', '_')}",
            output_dir="data",
            log_level="error",
            save_to_file=False,   # we'll save our own summary
        )
        tracker.start()
        fn()
        emissions_kg = tracker.stop()  # returns kgCO2e
    else:
        fn()
        emissions_kg = None

    t_elapsed = time.perf_counter() - t_start

    if emissions_kg is not None:
        print(f"  ✅ Emissions measured: {emissions_kg:.8f} kgCO2e")
        print(f"     ({emissions_kg * 1e6:.4f} mgCO2e  |  {emissions_kg * 1000:.6f} gCO2e)")
    else:
        print(f"  ⚠  Emissions: not measured (codecarbon unavailable)")

    print(f"  ⏱  Wall time:  {t_elapsed:.2f}s")

    return {
        "workload_name":  name,
        "description":    description,
        "emissions_kgco2e": round(emissions_kg, 9) if emissions_kg is not None else None,
        "wall_time_s":    round(t_elapsed, 3),
    }


def run_all_measurements() -> list[dict]:
    """
    Run all three tiers and return a list of result dicts.
    """
    print("=" * 55)
    print("  AI Misuse Evaluator — Emissions Measurement")
    print("  Comparing real measured values vs dataset estimates")
    print("=" * 55)

    measurements = [
        measure_workload(
            name="Tier1_Lightweight",
            fn=workload_tier1,
            description="500 x 512x512 matmul + softmax (small LLM inference)"
        ),
        measure_workload(
            name="Tier2_Medium",
            fn=workload_tier2,
            description="200 x 2048x2048 matmul + layer norm (BERT-size inference)"
        ),
        measure_workload(
            name="Tier3_Heavy",
            fn=workload_tier3,
            description="5s sustained 4096x4096 matmul (large model / training step)"
        ),
    ]
    return measurements


def print_comparison(measurements: list[dict]) -> None:
    """
    Print a comparison of measured values vs representative estimates
    from the usecases.csv dataset.
    """
    print("\n" + "=" * 70)
    print("  COMPARISON: Measured emissions vs dataset estimates")
    print("=" * 70)
    print(f"  {'Workload':<30} {'Measured kgCO2e':>16} {'Representative use case':>22}")
    print("  " + "-" * 68)

    comparisons = [
        ("Tier1_Lightweight", "Customer_Support_Bot (est. 0.000007)"),
        ("Tier2_Medium",      "Code_Assistant (est. 0.000008)"),
        ("Tier3_Heavy",       "Climate_Forecasting_Model (est. 0.000010)"),
    ]

    for name, use_case in comparisons:
        row = next((m for m in measurements if m["workload_name"] == name), None)
        if row and row["emissions_kgco2e"] is not None:
            val = f"{row['emissions_kgco2e']:.8f}"
        else:
            val = "not measured"
        print(f"  {name:<30} {val:>16}   {use_case}")

    print()
    print("  NOTE: The measured values above are for the isolated Python")
    print("  workload only. Real AI inference includes model loading, I/O,")
    print("  and preprocessing overheads not captured here. The estimates in")
    print("  usecases.csv are based on published benchmarks (Lacoste et al.,")
    print("  2019; Strubell et al., 2019) and scaled to per-query figures.")


def save_csv(measurements: list[dict], path: str = "data/emissions_measured.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["workload_name", "description",
                                                "emissions_kgco2e", "wall_time_s"])
        writer.writeheader()
        writer.writerows(measurements)
    print(f"\n✅ Measurements saved to {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all_measurements()
    print_comparison(results)
    save_csv(results)