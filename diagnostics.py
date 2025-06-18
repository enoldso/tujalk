"""
Diagnostics tool for Tujali Telehealth

This script helps diagnose and test the AI service and other critical
components of the Tujali Telehealth system.
"""

import os
import logging
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_huggingface_model():
    """
    Test the Hugging Face model loading and text generation
    """
    try:
        # Try to use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        model_name = "facebook/opt-350m"  # Using a smaller model for testing
        
        logger.info(f"Loading Hugging Face model: {model_name}")
        
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 0 else torch.float32
        )
        
        # Move model to GPU if available
        if device == 0:
            model = model.cuda()
        
        # Test text generation
        input_text = "Hello, I'm a medical assistant. "
        inputs = tokenizer(input_text, return_tensors="pt")
        
        if device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "success": True,
            "response": generated_text,
            "model": model_name,
            "device": "GPU" if device == 0 else "CPU"
        }
        
    except Exception as e:
        logger.error(f"Hugging Face model test error: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("\n=== Tujali Telehealth Diagnostics ===\n")
    
    print("\n--- Hugging Face Model Test ---")
    hf_result = test_huggingface_model()
    if hf_result["success"]:
        print(f"✅ Hugging Face model working on {hf_result['device']}! Response: {hf_result['response']}")
    else:
        print(f"❌ Hugging Face model error: {hf_result['error']}")