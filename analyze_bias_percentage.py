#!/usr/bin/env python3

import json
import sys
import numpy as np
from pathlib import Path

def analyze_prediction_bias(corpus_path, corpus_name):
    """Analyze prediction bias for GPT+Claude families from prediction frequency data."""
    
    data_file = corpus_path / "prediction_frequency_heatmap_data.json"
    
    if not data_file.exists():
        print(f"Warning: {data_file} not found")
        return None
    
    # Load the data
    with open(data_file, 'r') as f:
        data = json.load(f)

    matrix = np.array(data['frequency_matrix'])
    target_models = data['target_models']
    evaluator_models = data['evaluator_models']

    print(f'\n=== {corpus_name.upper()} CORPUS ANALYSIS ===')
    
    # Verify matrix structure
    print('Column sums (should be 1000 each):')
    col_sums = matrix.sum(axis=0)
    for col_idx, col_sum in enumerate(col_sums):
        model_name = target_models[col_idx].split('/')[-1]
        print(f'  {model_name}: {col_sum}')

    print('\nRow sums (evaluator predictions made):')
    row_sums = matrix.sum(axis=1)
    for row_idx, row_sum in enumerate(row_sums):
        model_name = evaluator_models[row_idx].split('/')[-1]
        print(f'  {model_name}: {row_sum}')

    # Identify GPT and Claude families by model name patterns
    gpt_indices = []
    claude_indices = []
    
    for i, model in enumerate(target_models):
        if 'openai/gpt' in model:
            gpt_indices.append(i)
        elif 'claude' in model:
            claude_indices.append(i)
    
    gpt_indices = np.array(gpt_indices)
    claude_indices = np.array(claude_indices)
    gpt_claude_indices = np.concatenate([gpt_indices, claude_indices])
    
    print(f'\nGPT family indices: {gpt_indices} = {[target_models[i].split("/")[-1] for i in gpt_indices]}')
    print(f'Claude family indices: {claude_indices} = {[target_models[i].split("/")[-1] for i in claude_indices]}')

    # Calculate predictions using numpy indexing
    # Sum all predictions TO GPT+Claude models (columns)
    gpt_claude_predictions_made = matrix[gpt_claude_indices, :].sum()
    total_predictions_made = matrix.sum()

    # Calculate overall percentage
    if total_predictions_made > 0:
        overall_pct = (gpt_claude_predictions_made / total_predictions_made) * 100
        print(f'\n{corpus_name} RESULTS:')
        print(f'  Total predictions: {total_predictions_made}')
        print(f'  GPT+Claude predictions: {gpt_claude_predictions_made}')
        print(f'  Percentage to GPT+Claude: {overall_pct:.1f}%')
        return overall_pct
    else:
        print(f'\n{corpus_name} RESULTS: No predictions made')
        return 0.0

def main():
    """Analyze both 100-word and 500-word corpora and compute average."""
    
    results_dir = Path("results")
    
    # Analyze both corpora
    pct_100 = analyze_prediction_bias(results_dir / "plot_100", "100-word")
    pct_500 = analyze_prediction_bias(results_dir / "plot_500", "500-word")
    
    # Calculate average
    percentages = [pct for pct in [pct_100, pct_500] if pct is not None]
    
    if percentages:
        avg_pct = sum(percentages) / len(percentages)
        print(f'\n=== SUMMARY ===')
        if pct_100 is not None:
            print(f'100-word corpus: {pct_100:.1f}%')
        if pct_500 is not None:
            print(f'500-word corpus: {pct_500:.1f}%')
        print(f'Average: {avg_pct:.1f}%')
    else:
        print('\n=== SUMMARY ===')
        print('No valid data found in either corpus')

if __name__ == "__main__":
    main()