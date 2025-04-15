# src/data_loader.py
import logging
from datasets import load_dataset
from transformers import AutoTokenizer

class DataLoader:
    """Handles dataset loading and preprocessing for text summarization."""
    
    def __init__(self, config):
        """
        Initialize the DataLoader with configuration.
        
        Args:
            config (dict): Configuration dictionary containing dataset and preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tokenizer = None
        self.dataset = None
        
    def load_dataset(self):
        """Load dataset from Hugging Face Datasets."""
        try:
            dataset_name = self.config['dataset']['name']
            self.logger.info(f"Loading dataset: {dataset_name}")
            self.dataset = load_dataset(dataset_name)
            self.logger.info(f"Dataset loaded successfully with {len(self.dataset['train'])} training samples")
            return self.dataset
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def initialize_tokenizer(self, model_name):
        """Initialize tokenizer for the specified model."""
        try:
            self.logger.info(f"Initializing tokenizer for model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            return self.tokenizer
        except Exception as e:
            self.logger.error(f"Error initializing tokenizer: {str(e)}")
            raise
            
    def preprocess_data(self, examples):
        """
        Preprocess data for the summarization model.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            dict: Processed examples with input_ids, attention_masks, and labels
        """
        input_column = self.config['dataset']['input_column']
        target_column = self.config['dataset']['target_column']
        
        inputs = examples[input_column]
        targets = examples[target_column]
        
        max_input_length = self.config['preprocessing']['max_input_length']
        max_target_length = self.config['preprocessing']['max_target_length']
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=max_input_length,
            padding="max_length",
            truncation=True
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=max_target_length,
                padding="max_length",
                truncation=True
            )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
        
    def prepare_dataset(self):
        """Prepare the dataset by applying preprocessing to all splits."""
        if self.dataset is None or self.tokenizer is None:
            raise ValueError("Dataset or tokenizer not initialized. Call load_dataset and initialize_tokenizer first.")
            
        self.logger.info("Preparing dataset with preprocessing")
        processed_dataset = self.dataset.map(
            self.preprocess_data,
            batched=True,
            # remove_columns=self.dataset["train"].column_names
        )
        
        return processed_dataset
