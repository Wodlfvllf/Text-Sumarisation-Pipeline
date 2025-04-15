# main.py
import argparse
import logging
import os
import yaml
from src.pipeline import SummarizationPipeline
from src.utils.logger import setup_logger
import json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text Summarization Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--text", type=str, 
                        help="Text to summarize (for single inference)")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save outputs")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main entry point for the summarization pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting summarization pipeline with config: {args.config}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = SummarizationPipeline(config)
    
    # Single text inference mode
    if args.text:
        logger.info("Running in single text inference mode")
        pipeline.initialize()
        summary = pipeline.run_inference(args.text)
        print(f"\nInput Text:\n{args.text}\n")
        print(f"Generated Summary:\n{summary[0]['summary_text']}\n")
        return
    
    # Full pipeline mode
    logger.info("Running in full pipeline mode")
    results = pipeline.run()
    
    # # Print evaluation results
    # if 'evaluation' in results:
    #     print("\nEvaluation Results:")
    #     for insight in results['evaluation']['analysis']['insights']:
    #         print(f"- {insight}")
        
    #     print("\nBest Example:")
    #     best = results['evaluation']['analysis']['examples']['best']
    #     print(f"Input: {best['input'][:100]}...")
    #     print(f"Reference: {best['reference']}")
    #     print(f"Prediction: {best['prediction']}")
    #     print(f"ROUGE-L: {best['rouge_scores']['rougeL']:.4f}")
        
    #     print("\nWorst Example:")
    #     worst = results['evaluation']['analysis']['examples']['worst']
    #     print(f"Input: {worst['input'][:100]}...")
    #     print(f"Reference: {worst['reference']}")
    #     print(f"Prediction: {worst['prediction']}")
    #     print(f"ROUGE-L: {worst['rouge_scores']['rougeL']:.4f}")
    # Print and save evaluation results
    if 'evaluation' in results:
        # Print results to console
        print("\nEvaluation Results:")
        for insight in results['evaluation']['analysis']['insights']:
            print(f"- {insight}")
        
        # Prepare output paths
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        examples_path = os.path.join(args.output_dir, "examples.json")
        examples_txt_path = os.path.join(args.output_dir, "examples.txt")
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': results['evaluation']['metrics'],
                'analysis': results['evaluation']['analysis']['insights']
            }, f, indent=2)
        
        # Save examples
        with open(examples_path, 'w') as f:
            json.dump(results['evaluation']['analysis']['examples'], f, indent=2)
        
        # Save human-readable examples
        with open(examples_txt_path, 'w') as f:
            best = results['evaluation']['analysis']['examples']['best']
            worst = results['evaluation']['analysis']['examples']['worst']
            
            f.write("=== Best Example ===\n")
            f.write(f"Input: {best['input']}\n")
            f.write(f"Reference: {best['reference']}\n")
            f.write(f"Prediction: {best['prediction']}\n")
            f.write(f"ROUGE-L: {best['rouge_scores']['rougeL']:.4f}\n\n")
            
            f.write("=== Worst Example ===\n")
            f.write(f"Input: {worst['input']}\n")
            f.write(f"Reference: {worst['reference']}\n")
            f.write(f"Prediction: {worst['prediction']}\n")
            f.write(f"ROUGE-L: {worst['rouge_scores']['rougeL']:.4f}\n")
        
        logger.info(f"\nSaved outputs to:")
        logger.info(f"- Metrics: {metrics_path}")
        logger.info(f"- Examples (JSON): {examples_path}")
        logger.info(f"- Examples (Text): {examples_txt_path}")

if __name__ == "__main__":
    main()
