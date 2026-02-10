import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

try:
    from .evaluator import EvaluationTask
except ImportError:
    from evaluator import EvaluationTask


class ResultsVisualizer:
    def __init__(self, style: str = "whitegrid"):
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def _get_short_model_name(self, model: str) -> str:
        """Get a shortened model name for display purposes."""
        # Remove provider prefix
        short_name = model.split('/')[-1]

        # Map to abbreviations like in plot_self_existence.py
        if 'kimi' in short_name.lower():
            return 'Kimi'
        elif 'gpt-4o-mini' in short_name.lower() or 'gpt-4.1-mini' in short_name.lower():
            return 'GPT4.1mini'
        elif 'gpt-4o' in short_name.lower() or 'gpt-4.1' in short_name.lower():
            return 'GPT4.1'
        elif 'gpt-5' in short_name.lower():
            return 'GPT5'
        elif 'claude' in short_name.lower():
            return 'Claude'
        elif 'qwen' in short_name.lower():
            return 'Qwen3'
        elif 'gemini' in short_name.lower():
            return 'Gemini'
        elif 'grok' in short_name.lower():
            return 'Grok'
        elif 'glm' in short_name.lower():
            return 'GLM'
        elif 'deepseek' in short_name.lower():
            return 'DeepSeek'

        return short_name
    
    def _save_plot_data(self, data: Dict, save_path: str) -> None:
        """Save plot data as JSON file alongside the plot.
        
        Args:
            data: Dictionary containing the plot data
            save_path: Path where the plot is saved (will create .json with same name)
        """
        import json
        if save_path:
            plot_path = Path(save_path)
            json_path = plot_path.parent / f"{plot_path.stem}_data.json"
            
            # Convert numpy arrays and other non-serializable types to Python types
            serializable_data = self._make_json_serializable(data)
            
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            print(f"Plot data saved to {json_path}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable types to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def create_accuracy_barplot(self, results: Dict[str, Dict[str, float]], 
                               title: str = None,  # Title parameter kept for compatibility but ignored
                               save_path: str = None,
                               task_type: EvaluationTask = EvaluationTask.EXACT_MODEL) -> plt.Figure:
        if task_type == EvaluationTask.BINARY_SELF:
            return self._create_binary_barplot(results, title, save_path)
        
        data = []
        
        for evaluator_model, accuracies in results.items():
            for target_model, accuracy in accuracies.items():
                if target_model != "overall":
                    data.append({
                        'evaluator_model': evaluator_model,
                        'target_model': target_model,
                        'accuracy': accuracy,
                        'is_self': evaluator_model == target_model
                    })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        evaluator_models = df['evaluator_model'].unique()
        target_models = df['target_model'].unique()
        
        x = np.arange(len(evaluator_models))
        width = 0.8 / len(target_models)
        
        colors = sns.color_palette("husl", len(target_models))
        
        for i, target_model in enumerate(target_models):
            target_data = df[df['target_model'] == target_model]
            accuracies = [target_data[target_data['evaluator_model'] == eval_model]['accuracy'].iloc[0] 
                         if len(target_data[target_data['evaluator_model'] == eval_model]) > 0 else 0
                         for eval_model in evaluator_models]
            
            bars = ax.bar(x + i * width - width * (len(target_models) - 1) / 2, 
                         accuracies, width, label=target_model, color=colors[i])
            
            for j, (bar, eval_model) in enumerate(zip(bars, evaluator_models)):
                if eval_model == target_model:
                    bar.set_edgecolor('red')
                    bar.set_linewidth(3)
                
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=20)
        
        ax.set_xlabel('Evaluator Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=20, fontweight='bold')
        # Title removed as requested
        ax.set_xticks(x)
        ax.set_xticklabels([model.split('/')[-1] for model in evaluator_models], 
                          rotation=45, ha='right', fontsize=20)
        ax.legend(title='Target Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
        ax.get_legend().get_title().set_fontsize(20)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelsize=20)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            
            # Save plot data
            plot_data = {
                'plot_type': 'accuracy_barplot',
                'task_type': task_type.value,
                'data': data,  # Raw data used to create the DataFrame
                'summary_stats': {
                    'evaluator_models': list(evaluator_models),
                    'target_models': list(target_models),
                    'accuracy_matrix': df.pivot(index='evaluator_model', columns='target_model', values='accuracy').to_dict()
                }
            }
            self._save_plot_data(plot_data, save_path)
        
        return fig
    
    def create_self_awareness_comparison(self, results: Dict[str, Dict[str, float]],
                                       title: str = None,  # Title parameter kept for compatibility but ignored
                                       save_path: str = None) -> plt.Figure:
        data = []
        
        for evaluator_model, accuracies in results.items():
            self_accuracy = accuracies.get(evaluator_model, 0)
            
            cross_accuracies = [acc for target_model, acc in accuracies.items() 
                              if target_model != evaluator_model and target_model != "overall"]
            avg_cross_accuracy = np.mean(cross_accuracies) if cross_accuracies else 0
            
            data.append({
                'model': evaluator_model,
                'self_identification': self_accuracy,
                'cross_identification': avg_cross_accuracy,
                'self_advantage': self_accuracy - avg_cross_accuracy
            })
        
        df = pd.DataFrame(data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, df['self_identification'], width, 
                       label='Self-identification', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, df['cross_identification'], width,
                       label='Cross-identification', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Model', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=20, fontweight='bold')
        # Title removed as requested
        ax1.set_xticks(x)
        ax1.set_xticklabels([self._get_short_model_name(model) for model in df['model']], 
                           rotation=45, ha='right', fontsize=20)
        ax1.legend(fontsize=20)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelsize=20)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=20)
        
        bars3 = ax2.bar(x, df['self_advantage'], color='green', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Model', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Self-Identification Advantage', fontsize=20, fontweight='bold')
        # Title removed as requested
        ax2.set_xticks(x)
        ax2.set_xticklabels([self._get_short_model_name(model) for model in df['model']], 
                           rotation=45, ha='right', fontsize=20)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelsize=20)
        
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.005 if height >= 0 else height - 0.015,
                    f'{height:.3f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=20)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def _create_binary_barplot(self, results: Dict[str, Dict[str, float]], 
                              title: str = None,  # Title parameter kept for compatibility but ignored
                              save_path: str = None) -> plt.Figure:
        data = []
        
        for evaluator_model, metrics in results.items():
            data.append({
                'model': evaluator_model,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0)
            })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(df))
        width = 0.2
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i * width - width * 1.5, df[metric], width, 
                         label=metric.replace('_', ' ').title(), color=colors[i])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=20)
        
        ax.set_xlabel('Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Score', fontsize=20, fontweight='bold')
        # Title removed as requested
        ax.set_xticks(x)
        ax.set_xticklabels([self._get_short_model_name(model) for model in df['model']], 
                          rotation=45, ha='right', fontsize=20)
        ax.legend(fontsize=20)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelsize=20)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Binary results plot saved to {save_path}")
        
        return fig
    
    def create_binary_confusion_matrix_plot(self, results: Dict[str, Dict[str, float]], 
                                           save_path: str = None) -> plt.Figure:
        """Create confusion matrix plots for binary classification results."""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, metrics) in enumerate(results.items()):
            # Extract confusion matrix values
            tp = metrics.get('true_positive', 0)
            fp = metrics.get('false_positive', 0)
            tn = metrics.get('true_negative', 0)
            fn = metrics.get('false_negative', 0)
            
            # Create confusion matrix
            confusion_matrix = np.array([[tp, fn], [fp, tn]])
            labels = [['True\nPositive', 'False\nNegative'], 
                     ['False\nPositive', 'True\nNegative']]
            
            ax = axes[idx]
            im = ax.imshow(confusion_matrix, cmap='Blues')
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, f'{labels[i][j]}\n{confusion_matrix[i, j]}',
                                 ha="center", va="center", color="black", fontsize=20)
            
            ax.set_xlabel('Predicted', fontsize=20, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=20, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Self', 'Other'], fontsize=20)
            ax.set_yticklabels(['Self', 'Other'], fontsize=20)
            
            # Add model name as subplot label
            model_short = model_name.split('/')[-1]
            ax.text(0.5, -0.15, model_short, ha='center', va='top', 
                   transform=ax.transAxes, fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Binary confusion matrix plot saved to {save_path}")
        
        return fig
    
    def create_binary_comparison_plot(self, results: Dict[str, Dict[str, float]], 
                                    save_path: str = None) -> plt.Figure:
        """Create a simple accuracy comparison plot for binary results."""
        data = []
        
        for model_name, metrics in results.items():
            data.append({
                'model': model_name,
                'accuracy': metrics.get('accuracy', 0)
            })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(df)), df['accuracy'], color='skyblue', alpha=0.8)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=20)
        
        ax.set_xlabel('Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Binary Classification Accuracy', fontsize=20, fontweight='bold')
        # Title removed as requested
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([self._get_short_model_name(model) for model in df['model']], 
                          rotation=45, ha='right', fontsize=20)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=20)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Binary comparison plot saved to {save_path}")
        
        return fig
    
    def print_summary_stats(self, results: Dict[str, Dict[str, float]], 
                           task_type: EvaluationTask = EvaluationTask.EXACT_MODEL) -> None:
        print("\n" + "="*60)
        if task_type == EvaluationTask.BINARY_SELF:
            print("BINARY SELF-IDENTIFICATION SUMMARY")
        else:
            print("EXACT MODEL PREDICTION SUMMARY")
        print("="*60)
        
        if task_type == EvaluationTask.BINARY_SELF:
            for evaluator_model, metrics in results.items():
                print(f"\n{evaluator_model}:")
                print("-" * 40)
                print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
                print(f"Precision: {metrics.get('precision', 0):.3f}")
                print(f"Recall: {metrics.get('recall', 0):.3f}")
                print(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
                print(f"True Positives: {metrics.get('true_positive', 0)}")
                print(f"False Positives: {metrics.get('false_positive', 0)}")
                print(f"True Negatives: {metrics.get('true_negative', 0)}")
                print(f"False Negatives: {metrics.get('false_negative', 0)}")
        else:
            for evaluator_model, accuracies in results.items():
                print(f"\n{evaluator_model}:")
                print("-" * 40)
                
                self_accuracy = accuracies.get(evaluator_model, 0)
                cross_accuracies = [acc for target_model, acc in accuracies.items() 
                                  if target_model != evaluator_model and target_model != "overall"]
                avg_cross_accuracy = np.mean(cross_accuracies) if cross_accuracies else 0
                overall_accuracy = accuracies.get("overall", 0)
                
                print(f"Overall accuracy: {overall_accuracy:.3f}")
                print(f"Self-identification: {self_accuracy:.3f}")
                print(f"Avg cross-identification: {avg_cross_accuracy:.3f}")
                print(f"Self-awareness advantage: {self_accuracy - avg_cross_accuracy:+.3f}")
        
        print("\n" + "="*60)
    
    def create_prediction_frequency_heatmap(self, predictions_data: List[Dict], 
                                          save_path: str = None) -> plt.Figure:
        """Create a heatmap showing how often model j predicts model i (frequency matrix).
        
        Args:
            predictions_data: List of prediction records from JSONL file
            save_path: Path to save the plot
        
        Returns:
            matplotlib Figure object
        """
        # Build frequency matrix: entry (i,j) = how often evaluator j predicts target i
        evaluators = sorted(set(pred['evaluator_model'] for pred in predictions_data))
        targets = sorted(set(pred['true_model'] for pred in predictions_data))
        
        # Initialize frequency matrix
        frequency_matrix = np.zeros((len(targets), len(evaluators)))
        
        # Count predictions
        for pred in predictions_data:
            if pred['task_type'] == 'exact_model' and 'predicted_model' in pred:
                evaluator_idx = evaluators.index(pred['evaluator_model'])
                predicted_model = pred.get('predicted_model', '')
                
                # Find index of predicted model in targets
                if predicted_model in targets:
                    target_idx = targets.index(predicted_model)
                    frequency_matrix[target_idx, evaluator_idx] += 1
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(len(evaluators) * 1.2 + 2, len(targets) * 1.0 + 2))
        
        # Create heatmap with custom colormap
        sns.heatmap(frequency_matrix, 
                   xticklabels=[self._get_short_model_name(model) for model in evaluators],
                   yticklabels=[model.split('/')[-1] for model in targets],
                   annot=True, fmt='.0f', 
                   cmap='Blues', 
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'Prediction Frequency'},
                   annot_kws={'fontsize': 20})
        
        # Customize the plot
        ax.set_xlabel('Evaluator Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Predicted Model', fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=20, rotation=45)
        ax.tick_params(axis='y', labelsize=20, rotation=0)
        
        # Customize colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Prediction Frequency', fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction frequency heatmap saved to {save_path}")
            
            # Save plot data
            plot_data = {
                'plot_type': 'prediction_frequency_heatmap',
                'frequency_matrix': frequency_matrix.tolist(),
                'evaluator_models': evaluators,
                'target_models': targets,
                'matrix_shape': frequency_matrix.shape,
                'total_predictions': len(predictions_data)
            }
            self._save_plot_data(plot_data, save_path)
        
        return fig
    
    def create_conditional_accuracy_heatmap(self, predictions_data: List[Dict], 
                                          save_path: str = None) -> plt.Figure:
        """Create a heatmap showing conditional accuracy: P(correct | evaluator=j, target=i).
        
        Args:
            predictions_data: List of prediction records from JSONL file
            save_path: Path to save the plot
        
        Returns:
            matplotlib Figure object
        """
        # Build conditional accuracy matrix: entry (i,j) = accuracy when evaluator j evaluates target i
        evaluators = sorted(set(pred['evaluator_model'] for pred in predictions_data))
        targets = sorted(set(pred['true_model'] for pred in predictions_data))
        
        # Initialize matrices for correct predictions and total predictions
        correct_matrix = np.zeros((len(targets), len(evaluators)))
        total_matrix = np.zeros((len(targets), len(evaluators)))
        
        # Count correct and total predictions
        for pred in predictions_data:
            if pred['task_type'] == 'exact_model':
                evaluator_idx = evaluators.index(pred['evaluator_model'])
                target_idx = targets.index(pred['true_model'])
                
                total_matrix[target_idx, evaluator_idx] += 1
                if pred.get('is_correct', False):
                    correct_matrix[target_idx, evaluator_idx] += 1
        
        # Calculate conditional accuracy (avoid division by zero)
        accuracy_matrix = np.where(total_matrix > 0, 
                                 correct_matrix / total_matrix, 
                                 np.nan)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(len(evaluators) * 1.2 + 2, len(targets) * 1.0 + 2))
        
        # Create heatmap with custom colormap
        mask = np.isnan(accuracy_matrix)  # Mask cells with no data
        
        sns.heatmap(accuracy_matrix,
                   xticklabels=[self._get_short_model_name(model) for model in evaluators],
                   yticklabels=[model.split('/')[-1] for model in targets],
                   annot=True, fmt='.3f',
                   cmap='RdYlGn', 
                   square=True,
                   linewidths=0.5,
                   vmin=0, vmax=1,
                   mask=mask,
                   cbar_kws={'label': 'Conditional Accuracy'},
                   annot_kws={'fontsize': 20})
        
        # Highlight diagonal (self-identification) with thick borders
        for i in range(min(len(targets), len(evaluators))):
            if targets[i] in evaluators:
                j = evaluators.index(targets[i])
                rect = plt.Rectangle((j, i), 1, 1, fill=False, 
                                   edgecolor='red', linewidth=4)
                ax.add_patch(rect)
        
        # Customize the plot  
        ax.set_xlabel('Evaluator Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Target Model (Ground Truth)', fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=20, rotation=45)
        ax.tick_params(axis='y', labelsize=20, rotation=0)
        
        # Customize colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Conditional Accuracy', fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Conditional accuracy heatmap saved to {save_path}")
            
            # Save plot data
            plot_data = {
                'plot_type': 'conditional_accuracy_heatmap',
                'accuracy_matrix': accuracy_matrix.tolist(),
                'evaluator_models': evaluators,
                'target_models': targets,
                'correct_counts': correct_matrix.tolist(),
                'total_counts': total_matrix.tolist(),
                'matrix_shape': accuracy_matrix.shape
            }
            self._save_plot_data(plot_data, save_path)
        
        return fig
    
    def create_empty_text_overall_plot(self, predictions_files: List[str], save_path: str = None) -> plt.Figure:
        """Create a plot showing overall empty predictions by task type."""
        import json
        
        data = []
        
        for predictions_file in predictions_files:
            if not Path(predictions_file).exists():
                print(f"Warning: File not found {predictions_file}")
                continue
                
            total_predictions = 0
            empty_predictions = 0
            
            try:
                with open(predictions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            pred = json.loads(line)
                            total_predictions += 1
                            
                            returned_text = pred.get('returned_text', '')
                            
                            # Check if returned_text is empty or invalid
                            if not returned_text or not returned_text.strip() or returned_text.startswith("ERROR:"):
                                empty_predictions += 1
                
                # Get task type from filename
                task_type = 'Binary Self' if 'binary_self' in predictions_file else 'Exact Model'
                
                # Add overall data
                data.append({
                    'task_type': task_type,
                    'total': total_predictions,
                    'empty': empty_predictions,
                    'empty_rate': empty_predictions / total_predictions if total_predictions > 0 else 0
                })
                    
                print(f"📊 {task_type}: {empty_predictions}/{total_predictions} empty predictions ({100*empty_predictions/total_predictions:.1f}%)")
                
            except Exception as e:
                print(f"Error reading {predictions_file}: {e}")
                continue
        
        if not data:
            print("No data to plot")
            return None
        
        df = pd.DataFrame(data)
        
        # Create single plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot overall empty counts by task type
        bars = ax.bar(df['task_type'], df['empty'], 
                     color=['#e74c3c', '#3498db'], alpha=0.7)
        ax.set_ylabel('Number of Empty Predictions', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Add value labels on bars (only fractions)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = df.iloc[i]['empty_rate'] * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + max(df['empty'])*0.01,
                    f'{percentage:.1f}%', 
                    ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Empty predictions overall plot saved to {save_path}")
        
        return fig
    
    def create_empty_text_by_model_plot(self, predictions_files: List[str], save_path: str = None) -> plt.Figure:
        """Create a plot showing empty prediction rates by evaluator model."""
        import json
        
        data = []
        
        for predictions_file in predictions_files:
            if not Path(predictions_file).exists():
                print(f"Warning: File not found {predictions_file}")
                continue
                
            evaluator_empty_counts = {}
            
            try:
                with open(predictions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            pred = json.loads(line)
                            
                            evaluator_model = pred.get('evaluator_model', 'Unknown')
                            returned_text = pred.get('returned_text', '')
                            
                            if evaluator_model not in evaluator_empty_counts:
                                evaluator_empty_counts[evaluator_model] = {'total': 0, 'empty': 0}
                            
                            evaluator_empty_counts[evaluator_model]['total'] += 1
                            
                            # Check if returned_text is empty or invalid
                            if not returned_text or not returned_text.strip() or returned_text.startswith("ERROR:"):
                                evaluator_empty_counts[evaluator_model]['empty'] += 1
                
                # Get task type from filename
                task_type = 'Binary Self' if 'binary_self' in predictions_file else 'Exact Model'
                
                # Add per-evaluator data
                for evaluator_model, counts in evaluator_empty_counts.items():
                    data.append({
                        'task_type': task_type,
                        'evaluator_model': evaluator_model,
                        'total': counts['total'],
                        'empty': counts['empty'],
                        'empty_rate': counts['empty'] / counts['total'] if counts['total'] > 0 else 0
                    })
                
            except Exception as e:
                print(f"Error reading {predictions_file}: {e}")
                continue
        
        if not data:
            print("No data to plot")
            return None
        
        df = pd.DataFrame(data)
        
        # Create single plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar plot
        task_types = df['task_type'].unique()
        evaluators = df['evaluator_model'].unique()
        
        x = np.arange(len(evaluators))
        width = 0.35
        
        colors = ['#e74c3c', '#3498db']
        
        for i, task_type in enumerate(task_types):
            task_subset = df[df['task_type'] == task_type]
            empty_rates = []
            for evaluator in evaluators:
                rate_data = task_subset[task_subset['evaluator_model'] == evaluator]
                if not rate_data.empty:
                    empty_rates.append(rate_data['empty_rate'].iloc[0] * 100)
                else:
                    empty_rates.append(0)
            
            bars = ax.bar(x + i*width, empty_rates, width, label=task_type, 
                          color=colors[i], alpha=0.7)
            
            # Add value labels (only percentages)
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.1f}%', 
                            ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        ax.set_ylabel('Empty Prediction Rate (%)', fontsize=24, fontweight='bold')
        ax.set_xlabel('Evaluator Model', fontsize=24, fontweight='bold')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels([self._get_short_model_name(model) for model in evaluators], rotation=45, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(fontsize=20)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Empty predictions by model plot saved to {save_path}")
        
        return fig
    
    def create_exact_model_self_prediction_plot(self, predictions_file: str, save_path: str = None) -> plt.Figure:
        """Create a plot showing the percentage each model predicts itself in exact model task."""
        import json
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        model_stats = {}
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        true_model = pred.get('true_model', '')
                        predicted_model = pred.get('predicted_model', '')
                        
                        if evaluator_model not in model_stats:
                            model_stats[evaluator_model] = {
                                'total_own_texts': 0,
                                'predicted_self': 0
                            }
                        
                        # Only count when evaluator is evaluating its own text
                        if evaluator_model == true_model:
                            model_stats[evaluator_model]['total_own_texts'] += 1
                            if predicted_model == evaluator_model:
                                model_stats[evaluator_model]['predicted_self'] += 1
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not model_stats:
            print("No self-evaluation data found")
            return None
        
        # Calculate percentages
        data = []
        for model, stats in model_stats.items():
            if stats['total_own_texts'] > 0:
                percentage = (stats['predicted_self'] / stats['total_own_texts']) * 100
                data.append({
                    'model': model,
                    'percentage': percentage,
                    'correct': stats['predicted_self'],
                    'total': stats['total_own_texts']
                })
        
        if not data:
            print("No data to plot")
            return None
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        bars = ax.bar([self._get_short_model_name(model) for model in df['model']], df['percentage'], 
                     color='#2ecc71', alpha=0.7)
        
        ax.set_ylabel('Accuracy on Its Own Generated Text (%)', fontsize=24, fontweight='bold')
        ax.set_xlabel('Evaluator Model', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(df['percentage'])*0.01,
                    f'{height:.1f}%', 
                    ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Exact model self-prediction plot saved to {save_path}")
        
        return fig
    
    def create_exact_model_prediction_bias_plot(self, predictions_file: str, save_path: str = None) -> plt.Figure:
        """Create a plot showing how often each model predicts itself across all evaluations (prediction bias)."""
        import json
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        # Collect prediction data
        prediction_data = {}
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        predicted_model = pred.get('predicted_model', '')
                        
                        if evaluator_model not in prediction_data:
                            prediction_data[evaluator_model] = {
                                'total_predictions': 0,
                                'self_predictions': 0
                            }
                        
                        prediction_data[evaluator_model]['total_predictions'] += 1
                        if predicted_model == evaluator_model:
                            prediction_data[evaluator_model]['self_predictions'] += 1
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not prediction_data:
            print("No prediction data found")
            return None
        
        # Calculate percentages
        data = []
        for model, stats in prediction_data.items():
            if stats['total_predictions'] > 0:
                percentage = (stats['self_predictions'] / stats['total_predictions']) * 100
                data.append({
                    'model': model,
                    'percentage': percentage,
                    'self_pred': stats['self_predictions'],
                    'total': stats['total_predictions']
                })
        
        if not data:
            print("No valid prediction bias data found")
            return None
        
        # Create DataFrame and sort by percentage
        df = pd.DataFrame(data)
        df['short_model'] = df['model'].apply(self._get_short_model_name)
        df = df.sort_values('percentage', ascending=True)
        
        # Create barplot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(df['short_model'], df['percentage'], color='coral', alpha=0.8)
        
        # Customize plot
        ax.set_ylabel('Evaluator Model', fontsize=36, fontweight='bold')
        ax.set_xlabel('Self-Prediction Bias (%)', fontsize=36, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(df['percentage'])*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', 
                    ha='left', va='center', fontsize=27, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Exact model prediction bias plot saved to {save_path}")
        
        return fig
    
    def create_exact_model_own_vs_other_accuracy_plot(self, predictions_file: str, save_path: str = None) -> plt.Figure:
        """Create a grouped bar plot comparing accuracy on own texts vs other texts."""
        import json
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        model_stats = {}
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        true_model = pred.get('true_model', '')
                        predicted_model = pred.get('predicted_model', '')
                        
                        if evaluator_model not in model_stats:
                            model_stats[evaluator_model] = {
                                'own_total': 0, 'own_correct': 0,
                                'other_total': 0, 'other_correct': 0
                            }
                        
                        if evaluator_model == true_model:
                            # Evaluating own text
                            model_stats[evaluator_model]['own_total'] += 1
                            if predicted_model == true_model:
                                model_stats[evaluator_model]['own_correct'] += 1
                        else:
                            # Evaluating other's text
                            model_stats[evaluator_model]['other_total'] += 1
                            if predicted_model == true_model:
                                model_stats[evaluator_model]['other_correct'] += 1
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not model_stats:
            print("No evaluation data found")
            return None
        
        # Calculate percentages
        data = []
        for model, stats in model_stats.items():
            own_acc = (stats['own_correct'] / stats['own_total'] * 100) if stats['own_total'] > 0 else 0
            other_acc = (stats['other_correct'] / stats['other_total'] * 100) if stats['other_total'] > 0 else 0
            
            data.append({
                'model': model,
                'own_accuracy': own_acc,
                'other_accuracy': other_acc,
                'own_correct': stats['own_correct'],
                'own_total': stats['own_total'],
                'other_correct': stats['other_correct'],
                'other_total': stats['other_total']
            })
        
        if not data:
            print("No valid accuracy data found")
            return None
        
        # Create DataFrame and sort by own accuracy
        df = pd.DataFrame(data)
        df['short_model'] = df['model'].apply(self._get_short_model_name)
        df = df.sort_values('own_accuracy', ascending=False)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['own_accuracy'], width, 
                       label='Accuracy on Own Text', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, df['other_accuracy'], width,
                       label='Accuracy on Other Text', color='#3498db', alpha=0.8)
        
        # Customize plot
        ax.set_ylabel('Accuracy (%)', fontsize=24, fontweight='bold')
        ax.set_xlabel('Evaluator Model', fontsize=24, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['short_model'], rotation=45, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(fontsize=18)
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Own text accuracy
            height1 = bar1.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 1,
                    f'{height1:.1f}%', 
                    ha='center', va='bottom', fontsize=16, fontweight='bold')
            
            # Other text accuracy
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 1,
                    f'{height2:.1f}%', 
                    ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Own vs other accuracy plot saved to {save_path}")
        
        return fig
    
    def create_binary_model_yes_prediction_plot(self, predictions_file: str, save_path: str = None) -> plt.Figure:
        """Create a plot showing the percentage each model predicts 'yes' (self-generated) in binary task."""
        import json
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        model_stats = {}
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        predicted_self = pred.get('predicted_self', False)
                        
                        if evaluator_model not in model_stats:
                            model_stats[evaluator_model] = {
                                'total_evaluations': 0,
                                'predicted_yes': 0
                            }
                        
                        model_stats[evaluator_model]['total_evaluations'] += 1
                        if predicted_self:
                            model_stats[evaluator_model]['predicted_yes'] += 1
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not model_stats:
            print("No binary evaluation data found")
            return None
        
        # Calculate percentages
        data = []
        for model, stats in model_stats.items():
            if stats['total_evaluations'] > 0:
                percentage = (stats['predicted_yes'] / stats['total_evaluations']) * 100
                data.append({
                    'model': model,
                    'percentage': percentage,
                    'yes_count': stats['predicted_yes'],
                    'total': stats['total_evaluations']
                })
        
        if not data:
            print("No data to plot")
            return None
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        bars = ax.bar([self._get_short_model_name(model) for model in df['model']], df['percentage'], 
                     color='#3498db', alpha=0.7)
        
        ax.set_ylabel('Yes Prediction Rate (%)', fontsize=36, fontweight='bold')
        ax.set_xlabel('Evaluator Model', fontsize=36, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(df['percentage'])*0.01,
                    f'{height:.1f}%', 
                    ha='center', va='bottom', fontsize=27, fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Binary model yes-prediction plot saved to {save_path}")
        
        return fig
    
    def create_exact_model_confusion_matrix(self, predictions_file: str, save_path: str = None) -> plt.Figure:
        """Create a confusion matrix showing the likelihood of model j predicting model i."""
        import json
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        # Collect prediction data
        prediction_data = {}
        all_models = set()
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        true_model = pred.get('true_model', '')
                        predicted_model = pred.get('predicted_model', '')
                        
                        if evaluator_model not in prediction_data:
                            prediction_data[evaluator_model] = {}
                        
                        if predicted_model not in prediction_data[evaluator_model]:
                            prediction_data[evaluator_model][predicted_model] = 0
                        
                        prediction_data[evaluator_model][predicted_model] += 1
                        all_models.add(evaluator_model)
                        all_models.add(predicted_model)
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not prediction_data:
            print("No prediction data found")
            return None
        
        # Convert to sorted lists for consistent ordering
        models = sorted(list(all_models))
        
        # Create confusion matrix (rows = evaluator models, columns = predicted models)
        # Entry (i,j) = likelihood of evaluator i predicting model j
        confusion_matrix = np.zeros((len(models), len(models)))
        
        for i, evaluator_model in enumerate(models):
            if evaluator_model in prediction_data:
                total_predictions = sum(prediction_data[evaluator_model].values())
                
                for j, predicted_model in enumerate(models):
                    count = prediction_data[evaluator_model].get(predicted_model, 0)
                    if total_predictions > 0:
                        confusion_matrix[i, j] = (count / total_predictions) * 100
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([model.split('/')[-1] for model in models], fontsize=18)
        ax.set_yticklabels([model.split('/')[-1] for model in models], fontsize=18)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(models)):
                value = confusion_matrix[i, j]
                if value > 0:
                    text = ax.text(j, i, f'{value:.1f}%',
                                 ha="center", va="center", color="white" if value > 50 else "black",
                                 fontsize=14, fontweight='bold')
        
        # Labels and formatting
        ax.set_xlabel('Predicted Model', fontsize=24, fontweight='bold')
        ax.set_ylabel('Evaluator Model', fontsize=24, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Prediction Likelihood (%)', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(labelsize=16)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Exact model confusion matrix saved to {save_path}")
        
        return fig
    
    def create_exact_model_accuracy_plot(self, predictions_file: str, save_path: str = None) -> plt.Figure:
        """Create a plot showing accuracy for each model in exact model task."""
        import json
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        model_stats = {}
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        is_correct = pred.get('is_correct', False)
                        
                        if evaluator_model not in model_stats:
                            model_stats[evaluator_model] = {'total': 0, 'correct': 0}
                        
                        model_stats[evaluator_model]['total'] += 1
                        if is_correct:
                            model_stats[evaluator_model]['correct'] += 1
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not model_stats:
            print("No accuracy data found")
            return None
        
        # Calculate accuracies
        data = []
        for model, stats in model_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                data.append({
                    'model': model,
                    'accuracy': accuracy,
                    'correct': stats['correct'],
                    'total': stats['total']
                })
        
        if not data:
            print("No data to plot")
            return None
        
        df = pd.DataFrame(data)
        
        # Create plot with style matching plot_self_existence.py
        plt.figure(figsize=(15, 6))

        # Create bars with consistent good-looking color
        model_names = [self._get_short_model_name(model) for model in df['model']]
        bars = plt.bar(range(len(model_names)), df['accuracy'], color='#9dc847')

        # Customize the chart
        plt.xlabel('Model', fontsize=24)
        plt.ylabel('Accuracy (%)', fontsize=24)
        plt.ylim(0, max(df['accuracy']) + 15)  # More space above max for labels
        plt.xlim(-0.5, len(model_names) - 0.5)

        # Set x-axis labels with rotation for readability
        plt.xticks(range(len(model_names)), model_names, rotation=0, ha='center', fontsize=20)
        plt.yticks(fontsize=28)
        plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

        # Add value labels on top of bars
        for i, (bar, acc) in enumerate(zip(bars, df['accuracy'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{acc:.0f}', ha='center', va='bottom', fontsize=26)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Exact model accuracy plot saved to {save_path}")

        return plt.gcf()
    
    def create_exact_model_f1_plot(self, predictions_file: str, save_path: str = None) -> plt.Figure:
        """Create a plot showing F1 scores for each model in exact model task (macro-averaged)."""
        import json
        from collections import defaultdict
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        # Collect all predictions by evaluator model
        evaluator_predictions = defaultdict(list)
        all_models = set()
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        true_model = pred.get('true_model', '')
                        predicted_model = pred.get('predicted_model', '')
                        
                        evaluator_predictions[evaluator_model].append({
                            'true': true_model,
                            'predicted': predicted_model
                        })
                        all_models.add(true_model)
                        all_models.add(predicted_model)
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not evaluator_predictions:
            print("No prediction data found")
            return None
        
        # Calculate macro-averaged F1 for each evaluator
        data = []
        target_models = sorted(list(all_models))
        
        for evaluator_model, predictions in evaluator_predictions.items():
            f1_scores = []
            
            for target_model in target_models:
                # Calculate precision, recall, and F1 for this target model
                tp = sum(1 for p in predictions if p['true'] == target_model and p['predicted'] == target_model)
                fp = sum(1 for p in predictions if p['true'] != target_model and p['predicted'] == target_model)
                fn = sum(1 for p in predictions if p['true'] == target_model and p['predicted'] != target_model)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f1_scores.append(f1)
            
            # Macro-averaged F1
            macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            
            data.append({
                'model': evaluator_model,
                'f1': macro_f1 * 100  # Convert to percentage
            })
        
        if not data:
            print("No data to plot")
            return None
        
        df = pd.DataFrame(data)
        
        # Create plot with style matching plot_self_existence.py
        plt.figure(figsize=(15, 6))

        # Create bars with consistent good-looking color
        model_names = [self._get_short_model_name(model) for model in df['model']]
        bars = plt.bar(range(len(model_names)), df['f1'], color='#9dc847')

        # Customize the chart
        plt.xlabel('Model', fontsize=24)
        plt.ylabel('F1 Score (%)', fontsize=24)
        plt.ylim(0, max(df['f1']) + 15)  # More space above max for labels
        plt.xlim(-0.5, len(model_names) - 0.5)

        # Set x-axis labels with rotation for readability
        plt.xticks(range(len(model_names)), model_names, rotation=0, ha='center', fontsize=20)
        plt.yticks(fontsize=28)
        plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

        # Add value labels on top of bars
        for i, (bar, f1) in enumerate(zip(bars, df['f1'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{f1:.0f}', ha='center', va='bottom', fontsize=26)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Exact model F1 score plot saved to {save_path}")

        return plt.gcf()
    
    def create_binary_model_accuracy_plot(self, predictions_file: str, save_path: str = None) -> plt.Figure:
        """Create a plot showing accuracy for each model in binary self task."""
        import json
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        model_stats = {}
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        is_correct = pred.get('is_correct', False)
                        
                        if evaluator_model not in model_stats:
                            model_stats[evaluator_model] = {'total': 0, 'correct': 0}
                        
                        model_stats[evaluator_model]['total'] += 1
                        if is_correct:
                            model_stats[evaluator_model]['correct'] += 1
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not model_stats:
            print("No accuracy data found")
            return None
        
        # Calculate accuracies
        data = []
        for model, stats in model_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                data.append({
                    'model': model,
                    'accuracy': accuracy,
                    'correct': stats['correct'],
                    'total': stats['total']
                })
        
        if not data:
            print("No data to plot")
            return None
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        bars = ax.bar([self._get_short_model_name(model) for model in df['model']], df['accuracy'], 
                     color='#3498db', alpha=0.7)
        
        ax.set_ylabel('Accuracy (%)', fontsize=36, fontweight='bold')
        ax.set_xlabel('Evaluator Model', fontsize=36, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(df['accuracy'])*0.01,
                    f'{height:.1f}%', 
                    ha='center', va='bottom', fontsize=27, fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Binary model accuracy plot saved to {save_path}")
        
        return fig
    
    def create_binary_model_f1_plot(self, predictions_file: str, save_path: str = None) -> plt.Figure:
        """Create a plot showing F1 scores for each model in binary self task."""
        import json
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        model_stats = {}
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        true_model = pred.get('true_model', '')
                        predicted_self = pred.get('predicted_self', False)
                        
                        if evaluator_model not in model_stats:
                            model_stats[evaluator_model] = {
                                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
                            }
                        
                        # Determine if this was actually a self-generated text
                        is_actually_self = (true_model == evaluator_model)
                        
                        # Calculate confusion matrix values
                        if predicted_self and is_actually_self:
                            model_stats[evaluator_model]['tp'] += 1
                        elif predicted_self and not is_actually_self:
                            model_stats[evaluator_model]['fp'] += 1
                        elif not predicted_self and not is_actually_self:
                            model_stats[evaluator_model]['tn'] += 1
                        elif not predicted_self and is_actually_self:
                            model_stats[evaluator_model]['fn'] += 1
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not model_stats:
            print("No binary classification data found")
            return None
        
        # Calculate F1 scores
        data = []
        for model, stats in model_stats.items():
            tp, fp, tn, fn = stats['tp'], stats['fp'], stats['tn'], stats['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            data.append({
                'model': model,
                'f1': f1 * 100  # Convert to percentage
            })
        
        if not data:
            print("No data to plot")
            return None
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        bars = ax.bar([self._get_short_model_name(model) for model in df['model']], df['f1'], 
                     color='#f39c12', alpha=0.7)
        
        ax.set_ylabel('F1 Score (%)', fontsize=36, fontweight='bold')
        ax.set_xlabel('Evaluator Model', fontsize=36, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(df['f1'])*0.01,
                    f'{height:.1f}%', 
                    ha='center', va='bottom', fontsize=27, fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Binary model F1 score plot saved to {save_path}")
            
            # Save plot data
            plot_data = {
                'plot_type': 'binary_model_f1_plot',
                'predictions_file': predictions_file,
                'raw_stats': model_stats,
                'computed_metrics': {
                    'model_scores': dict(zip(df['model'], df['f1']))
                },
                'summary_stats': {
                    'mean_f1': df['f1'].mean(),
                    'std_f1': df['f1'].std(),
                    'min_f1': df['f1'].min(),
                    'max_f1': df['f1'].max(),
                    'models_evaluated': len(df)
                }
            }
            self._save_plot_data(plot_data, save_path)
        
        return fig
    
    def create_prediction_network_graph(self, predictions_file: str, threshold: float = 3.0, save_path: str = None) -> plt.Figure:
        """Create a network graph where nodes are models and edges show prediction patterns above threshold."""
        import json
        import networkx as nx
        from collections import defaultdict
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        # Collect prediction data
        prediction_data = defaultdict(lambda: defaultdict(int))
        model_totals = defaultdict(int)
        all_models = set()
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        predicted_model = pred.get('predicted_model', '')
                        
                        if evaluator_model and predicted_model:
                            prediction_data[evaluator_model][predicted_model] += 1
                            model_totals[evaluator_model] += 1
                            all_models.add(evaluator_model)
                            all_models.add(predicted_model)
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not prediction_data:
            print("No prediction data found")
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add all models as nodes
        models = sorted(list(all_models))
        for model in models:
            # Use short model names for display
            short_name = self._get_short_model_name(model)
            G.add_node(short_name)
        
        # Add edges for predictions above threshold
        edge_data = []
        for evaluator_model in models:
            evaluator_short = self._get_short_model_name(evaluator_model)
            if evaluator_model in prediction_data:
                total_predictions = model_totals[evaluator_model]
                
                for predicted_model in models:
                    predicted_short = self._get_short_model_name(predicted_model)
                    count = prediction_data[evaluator_model].get(predicted_model, 0)
                    
                    if total_predictions > 0:
                        percentage = (count / total_predictions) * 100
                        
                        # Add edge if above threshold
                        if percentage > threshold:
                            G.add_edge(evaluator_short, predicted_short, 
                                     weight=percentage, count=count, total=total_predictions)
                            edge_data.append({
                                'from': evaluator_short,
                                'to': predicted_short,
                                'percentage': percentage,
                                'count': count
                            })
        
        if G.number_of_edges() == 0:
            print(f"No prediction patterns above {threshold}% threshold found")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Use spring layout for better node positioning
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue',
                              node_size=2000,
                              alpha=0.8,
                              ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=16,
                               font_weight='bold',
                               ax=ax)
        
        # Draw edges with varying thickness based on prediction percentage
        edges = G.edges(data=True)
        edge_weights = [edge[2]['weight'] for edge in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        
        # Normalize edge weights for visualization
        edge_widths = [(weight / max_weight) * 5 + 1 for weight in edge_weights]
        
        # Draw edges
        nx.draw_networkx_edges(G, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              width=edge_widths,
                              alpha=0.7,
                              ax=ax)
        
        # Add edge labels for percentages
        edge_labels = {}
        for edge in edges:
            from_node, to_node, data = edge
            percentage = data['weight']
            edge_labels[(from_node, to_node)] = f'{percentage:.1f}%'
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                    font_size=12,
                                    font_weight='bold',
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                             facecolor='white', 
                                             alpha=0.8),
                                    ax=ax)
        
        # Formatting
        ax.set_title(f'Model Prediction Network (>{threshold}% threshold)', 
                    fontsize=24, fontweight='bold', pad=20)
        ax.axis('off')  # Remove axes
        
        # Add legend
        legend_text = (f"• Nodes: AI Models\n"
                      f"• Edges: A→B means A predicts B >{threshold}% of time\n"
                      f"• Edge thickness: Prediction frequency\n"
                      f"• Edge labels: Exact percentages")
        
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
               fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Print network statistics
        print(f"\n📊 Network Statistics:")
        print(f"   • Nodes (models): {G.number_of_nodes()}")
        print(f"   • Edges (prediction patterns >{threshold}%): {G.number_of_edges()}")
        print(f"   • Average out-degree: {G.number_of_edges() / G.number_of_nodes():.1f}")
        
        # Print strongest prediction patterns
        if edge_data:
            edge_data.sort(key=lambda x: x['percentage'], reverse=True)
            print(f"\n🔝 Strongest prediction patterns:")
            for i, edge in enumerate(edge_data[:5]):  # Top 5
                print(f"   {i+1}. {edge['from']} → {edge['to']}: {edge['percentage']:.1f}% ({edge['count']} times)")
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPrediction network graph saved to {save_path}")
        
        return fig
    
    def create_prediction_network_graph_clean(self, predictions_file: str, threshold: float = 3.0, save_path: str = None) -> plt.Figure:
        """Create a clean network graph without title, legend, or edge labels."""
        import json
        import networkx as nx
        from collections import defaultdict
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        # Collect prediction data
        prediction_data = defaultdict(lambda: defaultdict(int))
        model_totals = defaultdict(int)
        all_models = set()
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        predicted_model = pred.get('predicted_model', '')
                        
                        if evaluator_model and predicted_model:
                            prediction_data[evaluator_model][predicted_model] += 1
                            model_totals[evaluator_model] += 1
                            all_models.add(evaluator_model)
                            all_models.add(predicted_model)
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not prediction_data:
            print("No prediction data found")
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add all models as nodes
        models = sorted(list(all_models))
        for model in models:
            # Use short model names for display
            short_name = self._get_short_model_name(model)
            G.add_node(short_name)
        
        # Add edges for predictions above threshold
        edge_data = []
        for evaluator_model in models:
            evaluator_short = self._get_short_model_name(evaluator_model)
            if evaluator_model in prediction_data:
                total_predictions = model_totals[evaluator_model]
                
                for predicted_model in models:
                    predicted_short = self._get_short_model_name(predicted_model)
                    count = prediction_data[evaluator_model].get(predicted_model, 0)
                    
                    if total_predictions > 0:
                        percentage = (count / total_predictions) * 100
                        
                        # Add edge if above threshold
                        if percentage > threshold:
                            G.add_edge(evaluator_short, predicted_short, 
                                     weight=percentage, count=count, total=total_predictions)
                            edge_data.append({
                                'from': evaluator_short,
                                'to': predicted_short,
                                'percentage': percentage,
                                'count': count
                            })
        
        if G.number_of_edges() == 0:
            print(f"No prediction patterns above {threshold}% threshold found")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Use spring layout for better node positioning
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue',
                              node_size=2000,
                              alpha=0.8,
                              ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=16,
                               font_weight='bold',
                               ax=ax)
        
        # Draw edges with varying thickness based on prediction percentage
        edges = G.edges(data=True)
        edge_weights = [edge[2]['weight'] for edge in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        
        # Normalize edge weights for visualization
        edge_widths = [(weight / max_weight) * 5 + 1 for weight in edge_weights]
        
        # Draw edges (no labels)
        nx.draw_networkx_edges(G, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              width=edge_widths,
                              alpha=0.7,
                              ax=ax)
        
        # Clean formatting - no title, no legend, no edge labels
        ax.axis('off')  # Remove axes
        
        plt.tight_layout()
        
        # Print network statistics (for console output)
        print(f"\n📊 Clean Network Statistics:")
        print(f"   • Nodes (models): {G.number_of_nodes()}")
        print(f"   • Edges (prediction patterns >{threshold}%): {G.number_of_edges()}")
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nClean prediction network graph saved to {save_path}")
        
        return fig
    
    def create_prediction_network_graph_with_logos(self, predictions_file: str, threshold: float = 3.0, save_path: str = None) -> plt.Figure:
        """Create a network graph with model logos as nodes."""
        import json
        import networkx as nx
        from collections import defaultdict
        import matplotlib.image as mpimg
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        
        if not Path(predictions_file).exists():
            print(f"Warning: File not found {predictions_file}")
            return None
        
        # Collect prediction data
        prediction_data = defaultdict(lambda: defaultdict(int))
        model_totals = defaultdict(int)
        all_models = set()
        
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pred = json.loads(line)
                        
                        evaluator_model = pred.get('evaluator_model', 'Unknown')
                        predicted_model = pred.get('predicted_model', '')
                        
                        if evaluator_model and predicted_model:
                            prediction_data[evaluator_model][predicted_model] += 1
                            model_totals[evaluator_model] += 1
                            all_models.add(evaluator_model)
                            all_models.add(predicted_model)
        
        except Exception as e:
            print(f"Error reading {predictions_file}: {e}")
            return None
        
        if not prediction_data:
            print("No prediction data found")
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add all models as nodes
        models = sorted(list(all_models))
        for model in models:
            short_name = self._get_short_model_name(model)
            G.add_node(short_name)
        
        # Add edges for predictions above threshold
        edge_data = []
        for evaluator_model in models:
            evaluator_short = self._get_short_model_name(evaluator_model)
            if evaluator_model in prediction_data:
                total_predictions = model_totals[evaluator_model]
                
                for predicted_model in models:
                    predicted_short = self._get_short_model_name(predicted_model)
                    count = prediction_data[evaluator_model].get(predicted_model, 0)
                    
                    if total_predictions > 0:
                        percentage = (count / total_predictions) * 100
                        
                        # Add edge if above threshold
                        if percentage > threshold:
                            G.add_edge(evaluator_short, predicted_short, 
                                     weight=percentage, count=count, total=total_predictions)
                            edge_data.append({
                                'from': evaluator_short,
                                'to': predicted_short,
                                'percentage': percentage,
                                'count': count
                            })
        
        if G.number_of_edges() == 0:
            print(f"No prediction patterns above {threshold}% threshold found")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 16))
        
        # Create custom circular layout with GPT models grouped together
        import math
        
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        # Define the desired order: GPT models together, then others
        gpt_models = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-5']
        other_models = [node for node in nodes if node not in gpt_models]
        
        # Arrange nodes in circle: GPT models first, then others
        ordered_nodes = gpt_models + other_models
        
        # Create circular positions
        pos = {}
        radius = 1.0
        for i, node in enumerate(ordered_nodes):
            angle = 2 * math.pi * i / n_nodes
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            pos[node] = (x, y)
        
        # Model logo file mapping (updated with manually downloaded logos)
        model_logo_files = {
            'gpt-4.1': 'openai.png',
            'gpt-4.1-mini': 'openai.png',
            'gpt-5': 'openai.png',
            'claude-sonnet-4': 'claude.png',
            'deepseek-v3': 'deepseek.png',
            'qwen3-325b': 'qwen.png',
            'qwen3-235b': 'qwen.png',
            'glm-4.5': 'zhipu.png',
            'grok-4': 'x.png',
            'kimi-k2': 'kimi.png',
            'gemini-2.5-flash': 'gemini.png',
        }
        
        # Individual logo size mapping (zoom values)
        model_logo_sizes = {
            'gpt-4.1': 0.48,          # GPT models: quadruple base size (0.12 * 4)
            'gpt-4.1-mini': 0.48,     # GPT models: quadruple base size
            'gpt-5': 0.48,            # GPT models: quadruple base size
            'claude-sonnet-4': 0.24,  # Claude: double base size
            'deepseek-v3': 0.12,      # DeepSeek: base size
            'qwen3-325b': 0.045,      # Qwen: 1.5x quarter size (0.03 * 1.5)
            'qwen3-235b': 0.045,      # Qwen: 1.5x quarter size
            'glm-4.5': 0.48,          # GLM: quadruple base size (0.12 * 4)
            'grok-4': 0.06,           # Grok: half base size
            'kimi-k2': 0.24,          # Kimi: double base size
            'gemini-2.5-flash': 0.24, # Gemini: double base size
        }
        
        # Emoji fallbacks for models without logos
        model_emoji_fallbacks = {
            'gpt-4.1': '🧠',
            'gpt-4.1-mini': '⚡',
            'claude-sonnet-4': '🎭',
            'deepseek-v3': '🌊',
            'qwen3-325b': '🏮',
            'glm-4.5': '⭐',
            'grok-4': '🚀',
            'llama-3.3': '🦙',
            'mistral-7b': '🌪️',
            'gemini-pro': '♊'
        }
        
        # Add logo images or fallback emojis for each node first
        logos_dir = Path("results/logos")
        successful_logos = 0
        
        for node, (x, y) in pos.items():
            logo_filename = model_logo_files.get(node)
            logo_loaded = False
            
            if logo_filename:
                logo_path = logos_dir / logo_filename
                if logo_path.exists():
                    try:
                        # Load and display logo image
                        img = mpimg.imread(logo_path)
                        
                        # Add transparency to the logo for better edge visibility
                        if img.shape[2] == 3:  # RGB image, add alpha channel
                            alpha = 0.8  # 80% opacity
                            img_with_alpha = np.zeros((img.shape[0], img.shape[1], 4))
                            img_with_alpha[:, :, :3] = img
                            img_with_alpha[:, :, 3] = alpha
                            img = img_with_alpha
                        elif img.shape[2] == 4:  # RGBA image, modify existing alpha
                            img[:, :, 3] = img[:, :, 3] * 0.8  # Reduce alpha to 80%
                        
                        # Use individual logo size for each model
                        logo_size = model_logo_sizes.get(node, 0.12)  # Default to 0.12 if not specified
                        imagebox = OffsetImage(img, zoom=logo_size)
                        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
                        ab.set_zorder(1)  # Set logo z-order to 1 (behind edges)
                        ax.add_artist(ab)
                        logo_loaded = True
                        successful_logos += 1
                    except Exception as e:
                        print(f"Warning: Could not load logo {logo_path}: {e}")
            
            # Fallback to emoji if logo couldn't be loaded
            if not logo_loaded:
                # Draw white circle background for emoji
                circle = plt.Circle((x, y), 0.05, color='white', alpha=0.9, zorder=2)
                ax.add_patch(circle)
                
                # Add emoji
                emoji = model_emoji_fallbacks.get(node, '🤖')
                ax.text(x, y, emoji, fontsize=105, ha='center', va='center', 
                       weight='bold', zorder=3)
            
            # Add text label with custom positioning for each model
            models_above = []
            models_right = ['gpt-4.1', 'gpt-5', 'gpt-4.1-mini']
            models_left = ['claude-sonnet-4']
            models_below_far = ['glm-4.5', 'qwen3-235b', 'deepseek-v3']  # Models needing extra spacing
            
            if node in models_above:
                # Place text above logo, move up more to avoid overlap
                text_x, text_y = x, y + 0.12  # Increased from 0.08 to 0.12
                ha_align, va_align = 'center', 'bottom'
            elif node in models_right:
                # Place text to the right of logo
                text_x, text_y = x + 0.12, y
                ha_align, va_align = 'left', 'center'
            elif node in models_left:
                # Place text to the left of logo
                text_x, text_y = x - 0.12, y
                ha_align, va_align = 'right', 'center'
            elif node in models_below_far:
                # Place text further below logo to avoid overlap with larger logos
                text_x, text_y = x, y - 0.12  # Increased from 0.08 to 0.12
                ha_align, va_align = 'center', 'top'
            else:
                # Place text below logo (default)
                text_x, text_y = x, y - 0.08
                ha_align, va_align = 'center', 'top'
            
            ax.text(text_x, text_y, node, fontsize=42, ha=ha_align, va=va_align, 
                   weight='bold', alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Draw edges on top of logos with variable arrow sizes
        edges = G.edges(data=True)
        if edges:
            edge_weights = [edge[2]['weight'] for edge in edges]
            max_weight = max(edge_weights) if edge_weights else 1
            
            # Draw edges individually with weight-based arrow sizing
            for (u, v, data) in edges:
                weight = data['weight']
                normalized_weight = weight / max_weight
                
                # Calculate edge width and arrow size
                edge_width = normalized_weight * 12 + 3  # Doubled width
                # Bigger arrows: range from 60 (thick) to 45 (thin) - 3x bigger
                arrow_size = 60 - (normalized_weight * 15)
                
                # Draw individual edge
                edge_collection = nx.draw_networkx_edges(G, pos,
                                      edgelist=[(u, v)],
                                      edge_color='#2E3440',  # Dark blue-gray
                                      arrows=True,
                                      arrowsize=arrow_size,  # Variable arrow size
                                      arrowstyle='-|>',  # Elegant arrow style
                                      width=edge_width,
                                      alpha=0.8,  # High opacity for visibility
                                      connectionstyle="arc3,rad=0.1",  # Subtle curve
                                      ax=ax)
                
                # Set z-order for each edge collection
                if edge_collection:
                    if isinstance(edge_collection, list):
                        for collection in edge_collection:
                            collection.set_zorder(2)
                    else:
                        edge_collection.set_zorder(2)
        
        # Clean formatting
        ax.axis('off')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        plt.tight_layout()
        
        # Print statistics
        print(f"\n📊 Logo Network Statistics:")
        print(f"   • Nodes (models): {G.number_of_nodes()}")
        print(f"   • Edges (prediction patterns >{threshold}%): {G.number_of_edges()}")
        print(f"   • Real logos loaded: {successful_logos}/{len(pos)}")
        print(f"   • Emoji fallbacks: {len(pos) - successful_logos}")
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nLogo prediction network graph saved to {save_path}")
        
        return fig
    
    def create_existence_matrix_heatmap(
        self, 
        existence_matrix: Dict[str, Dict[str, float]], 
        save_path: str = None,
        title: str = None
    ):
        """Create a heatmap showing the existence awareness matrix between models.
        
        Args:
            existence_matrix: Dictionary mapping evaluator -> {target -> existence_score}
            save_path: Path to save the plot
            title: Custom title for the plot
        """
        # Convert to DataFrame for plotting
        models = sorted(set(list(existence_matrix.keys()) + 
                           [target for targets in existence_matrix.values() for target in targets.keys()]))
        
        # Create matrix
        matrix_data = []
        for evaluator in models:
            row = []
            for target in models:
                if evaluator in existence_matrix and target in existence_matrix[evaluator]:
                    row.append(existence_matrix[evaluator][target])
                else:
                    row.append(np.nan)
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data, index=models, columns=models)
        
        # Shorten model names for display
        short_names = [self._get_short_model_name(model) for model in models]
        df.index = short_names
        df.columns = short_names
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap with custom colormap
        mask = df.isna()
        sns.heatmap(df, 
                   annot=True, 
                   fmt='.2f',
                   mask=mask,
                   cmap='RdYlGn',
                   center=0.5,
                   vmin=0.0,
                   vmax=1.0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'Existence Score'},
                   ax=ax)
        
        # Customize appearance
        if title:
            ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        ax.set_xlabel('Target Model', fontsize=16, fontweight='bold')
        ax.set_ylabel('Evaluator Model', fontsize=16, fontweight='bold')
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Existence matrix heatmap saved to {save_path}")
        
        return fig
    
    def create_existence_summary_plots(
        self,
        existence_matrix: Dict[str, Dict[str, float]],
        save_dir: str = "results/existence_experiment/plots",
        data_file: str = "results/existence_experiment/existence_predictions.jsonl"
    ):
        """Create comprehensive visualization suite for existence experiment results.

        Args:
            existence_matrix: Dictionary mapping evaluator -> {target -> existence_score}
            save_dir: Directory to save all plots
            data_file: Path to JSONL file with raw existence data for computing confidence intervals
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Main existence matrix heatmap
        self.create_existence_matrix_heatmap(
            existence_matrix,
            save_path=save_path / "existence_matrix_heatmap.pdf",
            title="LLM Existence Awareness Matrix"
        )

        # Load raw data for confidence intervals
        raw_data = self._load_existence_data(data_file)

        # 2. Self-awareness scores (diagonal) with confidence intervals
        models = sorted(existence_matrix.keys())
        self_scores, self_errors = self._compute_self_existence_stats(raw_data, models)

        fig, ax = plt.subplots(figsize=(12, 6))
        short_names = [self._get_short_model_name(model) for model in models]
        bars = ax.bar(range(len(models)), self_scores, color='steelblue', alpha=0.7,
                     yerr=self_errors, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
        ax.set_xlabel('Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Self-Existence Score', fontsize=20, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, score, error) in enumerate(zip(bars, self_scores, self_errors)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.02,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path / "self_existence_scores.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Average existence scores per model with confidence intervals
        avg_scores, avg_errors = self._compute_avg_existence_stats(raw_data, models)

        fig, ax = plt.subplots(figsize=(12, 6))
        models_sorted = sorted(avg_scores.keys(), key=lambda x: avg_scores[x], reverse=True)
        scores_sorted = [avg_scores[model] for model in models_sorted]
        errors_sorted = [avg_errors[model] for model in models_sorted]
        short_names_sorted = [self._get_short_model_name(model) for model in models_sorted]

        bars = ax.bar(range(len(models_sorted)), scores_sorted, color='darkgreen', alpha=0.7,
                     yerr=errors_sorted, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
        ax.set_xlabel('Evaluator Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Average Existence Score', fontsize=20, fontweight='bold')
        ax.set_xticks(range(len(models_sorted)))
        ax.set_xticklabels(short_names_sorted, rotation=45, ha='right', fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, score, error) in enumerate(zip(bars, scores_sorted, errors_sorted)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.02,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path / "average_existence_scores.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Target existence scores with confidence intervals
        target_scores, target_errors = self._compute_target_existence_stats(raw_data, models)

        fig, ax = plt.subplots(figsize=(12, 6))
        models_sorted = sorted(target_scores.keys(), key=lambda x: target_scores[x], reverse=True)
        scores_sorted = [target_scores[model] for model in models_sorted]
        errors_sorted = [target_errors[model] for model in models_sorted]
        short_names_sorted = [self._get_short_model_name(model) for model in models_sorted]

        bars = ax.bar(range(len(models_sorted)), scores_sorted, color='darkorange', alpha=0.7,
                     yerr=errors_sorted, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
        ax.set_xlabel('Target Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Average Existence Recognition Score', fontsize=20, fontweight='bold')
        ax.set_xticks(range(len(models_sorted)))
        ax.set_xticklabels(short_names_sorted, rotation=45, ha='right', fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, score, error) in enumerate(zip(bars, scores_sorted, errors_sorted)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.02,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path / "target_existence_scores.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Existence experiment visualizations saved to {save_dir}")
        print(f"   • existence_matrix_heatmap.pdf - Full existence matrix")
        print(f"   • self_existence_scores.pdf - Self-awareness scores with 95% CI")
        print(f"   • average_existence_scores.pdf - Average scores with 95% CI")
        print(f"   • target_existence_scores.pdf - Target recognition scores with 95% CI")

    def _load_existence_data(self, data_file: str) -> List[Dict]:
        """Load raw existence data from JSONL file"""
        import json
        data = []
        if Path(data_file).exists():
            with open(data_file, 'r') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return data

    def _compute_confidence_interval(self, scores: List[float]) -> float:
        """Compute 95% confidence interval half-width"""
        import scipy.stats as stats
        if len(scores) <= 1:
            return 0.0
        mean = np.mean(scores)
        se = stats.sem(scores)  # Standard error
        ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=se)
        return (ci[1] - ci[0]) / 2  # Half-width of CI

    def _compute_self_existence_stats(self, raw_data: List[Dict], models: List[str]) -> Tuple[List[float], List[float]]:
        """Compute self-existence scores and 95% CI error bars"""
        self_scores = []
        self_errors = []

        for model in models:
            # Find all self-existence records for this model
            self_records = [
                record['existence_score'] for record in raw_data
                if (record.get('evaluator_model') == model and
                    record.get('target_model') == model and
                    record.get('existence_score') is not None)
            ]

            if self_records:
                mean_score = np.mean(self_records)
                error = self._compute_confidence_interval(self_records)
            else:
                mean_score = 0.0
                error = 0.0

            self_scores.append(mean_score)
            self_errors.append(error)

        return self_scores, self_errors

    def _compute_avg_existence_stats(self, raw_data: List[Dict], models: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute average existence scores and 95% CI error bars"""
        avg_scores = {}
        avg_errors = {}

        for evaluator in models:
            # Find all records where this model is the evaluator
            evaluator_records = [
                record['existence_score'] for record in raw_data
                if (record.get('evaluator_model') == evaluator and
                    record.get('existence_score') is not None)
            ]

            if evaluator_records:
                mean_score = np.mean(evaluator_records)
                error = self._compute_confidence_interval(evaluator_records)
            else:
                mean_score = 0.0
                error = 0.0

            avg_scores[evaluator] = mean_score
            avg_errors[evaluator] = error

        return avg_scores, avg_errors

    def _compute_target_existence_stats(self, raw_data: List[Dict], models: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute average existence scores by target model and 95% CI error bars"""
        target_scores = {}
        target_errors = {}

        for target in models:
            # Find all records where this model is the target
            target_records = [
                record['existence_score'] for record in raw_data
                if (record.get('target_model') == target and
                    record.get('existence_score') is not None)
            ]

            if target_records:
                mean_score = np.mean(target_records)
                error = self._compute_confidence_interval(target_records)
            else:
                mean_score = 0.0
                error = 0.0

            target_scores[target] = mean_score
            target_errors[target] = error

        return target_scores, target_errors

    def _create_existence_plots_with_errors(
        self,
        existence_matrix: Dict[str, Dict[str, float]],
        save_dir: str,
        data_file: str
    ):
        """Create existence bar plots with error bars and _errors suffix."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Load raw data for confidence intervals
        raw_data = self._load_existence_data(data_file)

        # Self-existence scores with confidence intervals
        models = sorted(existence_matrix.keys())
        self_scores, self_errors = self._compute_self_existence_stats(raw_data, models)

        fig, ax = plt.subplots(figsize=(12, 6))
        short_names = [self._get_short_model_name(model) for model in models]
        bars = ax.bar(range(len(models)), self_scores, color='steelblue', alpha=0.7,
                     yerr=self_errors, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
        ax.set_xlabel('Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Self-Existence Score', fontsize=20, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, score, error) in enumerate(zip(bars, self_scores, self_errors)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.02,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path / "self_existence_scores_errors.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Self-existence plot with errors saved: {save_path / 'self_existence_scores_errors.pdf'}")

        # Average existence scores with confidence intervals
        avg_scores, avg_errors = self._compute_avg_existence_stats(raw_data, models)

        fig, ax = plt.subplots(figsize=(12, 6))
        models_sorted = sorted(avg_scores.keys(), key=lambda x: avg_scores[x], reverse=True)
        scores_sorted = [avg_scores[model] for model in models_sorted]
        errors_sorted = [avg_errors[model] for model in models_sorted]
        short_names_sorted = [self._get_short_model_name(model) for model in models_sorted]

        bars = ax.bar(range(len(models_sorted)), scores_sorted, color='darkgreen', alpha=0.7,
                     yerr=errors_sorted, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
        ax.set_xlabel('Evaluator Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Average Existence Score', fontsize=20, fontweight='bold')
        ax.set_xticks(range(len(models_sorted)))
        ax.set_xticklabels(short_names_sorted, rotation=45, ha='right', fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, score, error) in enumerate(zip(bars, scores_sorted, errors_sorted)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.02,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path / "average_existence_scores_errors.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Average existence plot with errors saved: {save_path / 'average_existence_scores_errors.pdf'}")

        # Target existence scores with confidence intervals
        target_scores, target_errors = self._compute_target_existence_stats(raw_data, models)

        fig, ax = plt.subplots(figsize=(12, 6))
        models_sorted = sorted(target_scores.keys(), key=lambda x: target_scores[x], reverse=True)
        scores_sorted = [target_scores[model] for model in models_sorted]
        errors_sorted = [target_errors[model] for model in models_sorted]
        short_names_sorted = [self._get_short_model_name(model) for model in models_sorted]

        bars = ax.bar(range(len(models_sorted)), scores_sorted, color='darkorange', alpha=0.7,
                     yerr=errors_sorted, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
        ax.set_xlabel('Target Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Average Existence Recognition Score', fontsize=20, fontweight='bold')
        ax.set_xticks(range(len(models_sorted)))
        ax.set_xticklabels(short_names_sorted, rotation=45, ha='right', fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, score, error) in enumerate(zip(bars, scores_sorted, errors_sorted)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.02,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path / "target_existence_scores_errors.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Target existence plot with errors saved: {save_path / 'target_existence_scores_errors.pdf'}")