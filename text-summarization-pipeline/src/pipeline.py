# # src/pipeline.py
# import logging
# from .data_loader import DataLoader
# from .model import SummarizationModel
# from .evaluator import Evaluator

# class SummarizationPipeline:
#     """Main pipeline coordinating data loading, model inference, and evaluation."""
    
#     def __init__(self, config):
#         """
#         Initialize the summarization pipeline with configuration.
        
#         Args:
#             config (dict): Configuration dictionary
#         """
#         self.config = config
#         self.logger = logging.getLogger(__name__)
#         self.data_loader = DataLoader(config)
#         self.model = SummarizationModel(config)
#         self.evaluator = Evaluator(config)
        
#     def initialize(self):
#         """Initialize all components of the pipeline."""
#         self.logger.info("Initializing pipeline components")
        
#         # Load dataset
#         dataset = self.data_loader.load_dataset()
        
#         # Initialize tokenizer
#         model_name = self.config['model']['name']
#         tokenizer = self.data_loader.initialize_tokenizer(model_name)
        
#         # Load model
#         self.model.load_model(tokenizer)
        
#         # Prepare dataset if needed for evaluation
#         if self.config['pipeline']['run_evaluation']:
#             processed_dataset = self.data_loader.prepare_dataset()
#             return processed_dataset
        
#         return dataset
    
#     def run_inference(self, text):
#         """
#         Run inference on a single text or list of texts.
        
#         Args:
#             text (str or list): Text to summarize
            
#         Returns:
#             list: Generated summaries
#         """
#         self.logger.info("Running inference")
#         summaries = self.model.summarize(text)
#         return summaries
    
#     def run_batch_inference(self, dataset, split="test", num_samples=10):
#         """
#         Run inference on a batch of examples from the dataset.
        
#         Args:
#             dataset: Dataset to use
#             split (str): Dataset split to use
#             num_samples (int): Number of samples to process
            
#         Returns:
#             dict: Dictionary with inputs, references, and predictions
#         """
#         self.logger.info(f"Running batch inference on {num_samples} samples from {split} split")
        
#         input_column = self.config['dataset']['input_column']
#         target_column = self.config['dataset']['target_column']
        
#         # Select samples
#         samples = dataset[split].select(range(min(num_samples, len(dataset[split]))))
        
#         inputs = samples[input_column]
#         references = samples[target_column]
        
#         # Generate summaries
#         predictions = []
#         for text in inputs:
#             summary = self.model.summarize(text)
#             predictions.append(summary[0]['summary_text'])
        
#         return {
#             'inputs': inputs,
#             'references': references,
#             'predictions': predictions
#         }
    
#     def run_evaluation(self, results):
#         """
#         Run evaluation on inference results.
        
#         Args:
#             results (dict): Dictionary with inputs, references, and predictions
            
#         Returns:
#             dict: Evaluation metrics
#         """
#         self.logger.info("Running evaluation")
#         metrics = self.evaluator.calculate_metrics(
#             predictions=results['predictions'],
#             references=results['references']
#         )
        
#         # Generate analysis report
#         analysis = self.evaluator.analyze_results(
#             inputs=results['inputs'],
#             references=results['references'],
#             predictions=results['predictions'],
#             metrics=metrics
#         )
        
#         return {
#             'metrics': metrics,
#             'analysis': analysis
#         }
    
#     def run(self):
#         """
#         Run the complete summarization pipeline.
        
#         Returns:
#             dict: Results including metrics and analysis
#         """
#         self.logger.info("Running complete summarization pipeline")
        
#         # Initialize components
#         dataset = self.initialize()
        
#         # Run batch inference
#         num_samples = self.config['evaluation']['num_samples']
#         results = self.run_batch_inference(dataset, num_samples=num_samples)
        
#         # Run evaluation
#         if self.config['pipeline']['run_evaluation']:
#             evaluation_results = self.run_evaluation(results)
#             return {
#                 'results': results,
#                 'evaluation': evaluation_results
#             }
        
#         return {'results': results}

import logging
from .data_loader import DataLoader
from .model import SummarizationModel
from .evaluator import Evaluator

class SummarizationPipeline:
    """Main pipeline coordinating data flow and processing."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader(config)
        self.model = SummarizationModel(config)
        self.evaluator = Evaluator(config)
        
    def initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing pipeline components")
        
        # Load dataset and tokenizer
        self.data_loader.load_dataset()
        tokenizer = self.data_loader.initialize_tokenizer(
            self.config['model']['name']
        )
        
        # Load model
        self.model.load_model(tokenizer)
        
    def run_inference(self, text):
        """Run inference on single text."""
        self.logger.info("Running inference")
        return self.model.summarize(text)
    
    def run_batch_inference(self, split="test", num_samples=10):
        """Run batch inference on original dataset."""
        self.logger.info(f"Running batch inference on {num_samples} samples")
        
        input_column = self.config['dataset']['input_column']
        target_column = self.config['dataset']['target_column']
        
        # Access original dataset
        dataset = self.data_loader.dataset[split]
        samples = dataset.select(range(min(num_samples, len(dataset))))
        
        inputs = samples[input_column]
        references = samples[target_column]
        
        # Generate summaries
        predictions = []
        for text in inputs:
            summary = self.model.summarize(text)
            predictions.append(summary[0]['summary_text'])
        
        return {
            'inputs': inputs,
            'references': references,
            'predictions': predictions
        }
    
    def run_evaluation(self, results):
        """Run evaluation on results."""
        self.logger.info("Running evaluation")
        metrics = self.evaluator.calculate_metrics(
            results['predictions'], results['references']
        )
        analysis = self.evaluator.analyze_results(
            results['inputs'], results['references'], results['predictions'], metrics
        )
        return {'metrics': metrics, 'analysis': analysis}
    
    def run(self):
        """Execute complete pipeline."""
        self.logger.info("Starting pipeline execution")
        self.initialize()
        
        results = self.run_batch_inference(
            num_samples=self.config['evaluation']['num_samples']
        )
        
        if self.config['pipeline']['run_evaluation']:
            evaluation = self.run_evaluation(results)
            return {'results': results, 'evaluation': evaluation}
        
        return {'results': results}
