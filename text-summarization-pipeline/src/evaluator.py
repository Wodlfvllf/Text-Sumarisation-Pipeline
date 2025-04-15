# src/evaluator.py
import logging
import numpy as np
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import pandas as pd

class Evaluator:
    """Handles evaluation of summarization results using various metrics."""
    
    def __init__(self, config):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_metrics(self, predictions, references):
        """
        Calculate evaluation metrics for generated summaries.
        
        Args:
            predictions (list): Generated summary texts
            references (list): Reference summary texts
            
        Returns:
            dict: Dictionary of metrics
        """
        self.logger.info("Calculating evaluation metrics")
        
        # Calculate ROUGE scores
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            score = self.scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(score[key].fmeasure)
        
        # Calculate averages
        avg_scores = {key: np.mean(values) for key, values in rouge_scores.items()}
        
        # Calculate summary length statistics
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        length_stats = {
            'pred_length_avg': np.mean(pred_lengths),
            'pred_length_std': np.std(pred_lengths),
            'ref_length_avg': np.mean(ref_lengths),
            'ref_length_std': np.std(ref_lengths),
        }
        
        return {
            'rouge': avg_scores,
            'length_stats': length_stats,
            'individual_scores': rouge_scores
        }
    
    def analyze_results(self, inputs, references, predictions, metrics):
        """
        Analyze evaluation results and generate insights.
        
        Args:
            inputs (list): Input texts
            references (list): Reference summaries
            predictions (list): Generated summaries
            metrics (dict): Calculated metrics
            
        Returns:
            dict: Analysis results and insights
        """
        self.logger.info("Analyzing evaluation results")
        
        # Find best and worst examples based on ROUGE-L
        rouge_l_scores = metrics['individual_scores']['rougeL']
        best_idx = np.argmax(rouge_l_scores)
        worst_idx = np.argmin(rouge_l_scores)
        
        # Calculate correlation between input length and ROUGE scores
        input_lengths = [len(text.split()) for text in inputs]
        
        # Simple correlation analysis
        r1_corr = np.corrcoef(input_lengths, metrics['individual_scores']['rouge1'])[0, 1]
        r2_corr = np.corrcoef(input_lengths, metrics['individual_scores']['rouge2'])[0, 1]
        rl_corr = np.corrcoef(input_lengths, metrics['individual_scores']['rougeL'])[0, 1]
        
        # Prepare example comparisons
        examples = {
            'best': {
                'input': inputs[best_idx],
                'reference': references[best_idx],
                'prediction': predictions[best_idx],
                'rouge_scores': {
                    'rouge1': metrics['individual_scores']['rouge1'][best_idx],
                    'rouge2': metrics['individual_scores']['rouge2'][best_idx],
                    'rougeL': metrics['individual_scores']['rougeL'][best_idx]
                }
            },
            'worst': {
                'input': inputs[worst_idx],
                'reference': references[worst_idx],
                'prediction': predictions[worst_idx],
                'rouge_scores': {
                    'rouge1': metrics['individual_scores']['rouge1'][worst_idx],
                    'rouge2': metrics['individual_scores']['rouge2'][worst_idx],
                    'rougeL': metrics['individual_scores']['rougeL'][worst_idx]
                }
            }
        }
        
        # Generate insights
        insights = [
            f"Average ROUGE-1: {metrics['rouge']['rouge1']:.4f}",
            f"Average ROUGE-2: {metrics['rouge']['rouge2']:.4f}",
            f"Average ROUGE-L: {metrics['rouge']['rougeL']:.4f}",
            f"Average generated summary length: {metrics['length_stats']['pred_length_avg']:.1f} words",
            f"Average reference summary length: {metrics['length_stats']['ref_length_avg']:.1f} words",
            f"Correlation between input length and ROUGE-1: {r1_corr:.4f}",
            f"Correlation between input length and ROUGE-2: {r2_corr:.4f}",
            f"Correlation between input length and ROUGE-L: {rl_corr:.4f}"
        ]
        
        return {
            'insights': insights,
            'examples': examples,
            'correlations': {
                'rouge1_vs_length': r1_corr,
                'rouge2_vs_length': r2_corr,
                'rougeL_vs_length': rl_corr
            }
        }
    
    def visualize_results(self, metrics, output_path=None):
        """
        Create visualizations for evaluation results.
        
        Args:
            metrics (dict): Calculated metrics
            output_path (str, optional): Path to save visualizations
            
        Returns:
            dict: Paths to saved visualizations
        """
        self.logger.info("Creating visualizations for evaluation results")
        
        # Plot ROUGE score distributions
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, rouge_type in enumerate(['rouge1', 'rouge2', 'rougeL']):
            scores = metrics['individual_scores'][rouge_type]
            ax[i].hist(scores, bins=10, alpha=0.7)
            ax[i].set_title(f'{rouge_type} Score Distribution')
            ax[i].set_xlabel('Score')
            ax[i].set_ylabel('Frequency')
            ax[i].axvline(x=np.mean(scores), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(scores):.4f}')
            ax[i].legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path}/rouge_distribution.png")
            plt.close()
            
        return {"rouge_dist": f"{output_path}/rouge_distribution.png" if output_path else None}
