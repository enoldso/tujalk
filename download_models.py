"""
Script to download and cache Hugging Face models for offline use.
"""

import os
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    set_seed
)
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model(model_name, model_type="causal"):
    """
    Download and cache a Hugging Face model and tokenizer.
    
    Args:
        model_name (str): Name of the model on Hugging Face Hub
        model_type (str): Type of model (causal, sequence-classification, etc.)
    """
    try:
        logger.info(f"Downloading model: {model_name}")
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Download model
        logger.info("Downloading model weights...")
        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"Model moved to GPU: {next(model.parameters()).device}")
        
        # Test the model with a small inference
        logger.info("Testing model with sample inference...")
        inputs = tokenizer("Hello, I'm a medical assistant. ", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test generation successful. Sample output: {generated_text[:100]}...")
        
        # Save model and tokenizer locally
        local_path = os.path.join("models", model_name.replace("/", "_"))
        os.makedirs(local_path, exist_ok=True)
        
        logger.info(f"Saving model to {local_path}...")
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        
        logger.info(f"Successfully downloaded and saved {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {str(e)}")
        return False

def main():
    """Download all required models."""
    # List of models to download
    models_to_download = [
        "facebook/opt-350m",  # Small model for testing
        "facebook/opt-1.3b",  # Medium-sized model for better quality
    ]
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download each model
    for model_name in models_to_download:
        success = download_model(model_name)
        if success:
            logger.info(f"✅ Successfully processed {model_name}")
        else:
            logger.error(f"❌ Failed to process {model_name}")
    
    logger.info("Model download process completed!")

if __name__ == "__main__":
    main()
