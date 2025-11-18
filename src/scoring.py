import pandas as pd

def score_use_case(row):
    """
    Basic scoring engine for MVP.
    
    Environmental score = 10 - energy_kwh (lower energy = better)
    Ethical score       = 10 - (2 * ethical_risk)
    Creativity score    = 10 - (2 * creative_displacement)
    
    Final score is the average, bounded between 0 and 10.
    """
    # Component calculations
    environmental = 10 - row['energy_kwh']
    ethical = 10 - 2 * row['ethical_risk']
    creative = 10 - 2 * row['creative_displacement']
    
    # Average score
    final_score = (environmental + ethical + creative) / 3
    
    # Ensure value is between 0–10
    return max(min(final_score, 10), 0)


def evaluate_dataset(csv_path):
    """Loads dataset, applies scoring, and returns the DataFrame."""
    df = pd.read_csv(csv_path)
    df['responsibility_score'] = df.apply(score_use_case, axis=1)
    return df


if __name__ == "__main__":
    file = "data/sample_usecases.csv"
    print(f"Evaluating dataset: {file}")
    
    try:
        results = evaluate_dataset(file)
        print("\nEvaluation Results:")
        print(results)
        
        # Save results
        results.to_csv("data/results_scored.csv", index=False)
        print("\nScores saved to data/results_scored.csv")
    except FileNotFoundError:
        print(f"\nERROR: The file {file} was not found. Please create it first.")
