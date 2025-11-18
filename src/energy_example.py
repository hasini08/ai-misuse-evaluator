from codecarbon import EmissionsTracker

def dummy_task():
    """
    This simulates some computation work.
    You can replace it later with a real model or algorithm.
    """
    total = 0
    for i in range(10_000_00):  # 1 million iterations
        total += i
    return total

if __name__ == "__main__":
    tracker = EmissionsTracker()
    tracker.start()

    result = dummy_task()

    emissions_kg = tracker.stop()  # kg of CO2 equivalent

    print("Dummy task result:", result)
    print("Estimated emissions:", emissions_kg, "kg CO2eq")
