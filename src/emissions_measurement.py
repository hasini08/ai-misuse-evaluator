from codecarbon import EmissionsTracker
import time
import numpy as np

def run_dummy_ai_workload(duration_seconds=5):
    """
    Simulates AI workload (matrix multiplications).
    """
    start = time.time()
    while time.time() - start < duration_seconds:
        _ = np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)

def measure_emissions(duration_seconds=5):
    tracker = EmissionsTracker()
    tracker.start()

    run_dummy_ai_workload(duration_seconds)

    emissions = tracker.stop()
    return emissions

if __name__ == "__main__":
    value = measure_emissions(3)
    print(f"Measured emissions: {value:.6f} kgCO2e")