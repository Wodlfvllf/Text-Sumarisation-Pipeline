# src/model.py
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, pipeline

class SummarizationModel:
    """Handles loading and inference for text summarization models."""
    
    def __init__(self, config):
        """
        Initialize the summarization model with configuration.
        
        Args:
            config (dict): Configuration dictionary containing model parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.summarization_pipeline = None
        
    def load_model(self, tokenizer):
        """
        Load the pretrained summarization model.
        
        Args:
            tokenizer: Initialized tokenizer for the model
            
        Returns:
            model: Loaded model
        """
        try:
            model_name = self.config['model']['name']
            self.logger.info(f"Loading model: {model_name}")
            
            # Load the model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = tokenizer
            
            # Create summarization pipeline
            device = 0 if torch.cuda.is_available() else -1
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            
            self.logger.info(f"Model loaded successfully on device: {device}")
            return self.model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def summarize(self, text, **kwargs):
        """
        Generate summary for the given text.
        
        Args:
            text (str or list): Text to summarize
            **kwargs: Additional parameters for the summarization pipeline
            
        Returns:
            list: Generated summaries
        """
        if self.summarization_pipeline is None:
            raise ValueError("Model not initialized. Call load_model first.")
        
        # Set default parameters from config if not provided
        params = {
            'max_length': self.config['generation']['max_length'],
            'min_length': self.config['generation']['min_length'],
            'do_sample': self.config['generation']['do_sample'],
            'early_stopping': True
        }
        
        # Override with any provided kwargs
        params.update(kwargs)
        
        try:
            self.logger.info(f"Generating summary with parameters: {params}")
            summaries = self.summarization_pipeline(text, **params)
            return summaries
        except Exception as e:
            self.logger.error(f"Error during summarization: {str(e)}")
            raise
    
    def save_model(self, output_dir):
        """Save the model to the specified directory."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not initialized.")
            
        try:
            self.logger.info(f"Saving model to {output_dir}")
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self.logger.info(f"Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
