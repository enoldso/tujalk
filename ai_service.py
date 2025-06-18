"""
AI Service for Tujali Telehealth

This module provides AI-powered personalized health recommendations
using locally downloaded Hugging Face models. It generates health tips based on 
patient data, symptoms, and regional health concerns.
"""

import os
import json
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "facebook/opt-350m"  # Default model name
MODELS_DIR = Path("models")  # Local directory to store models
LOCAL_MODEL_PATH = MODELS_DIR / MODEL_NAME.replace("/", "_")

# Initialize the model and tokenizer
generator = None

def load_local_model():
    """Load the model from local storage."""
    global generator
    
    try:
        # Check if model exists locally
        if not LOCAL_MODEL_PATH.exists():
            logger.error(f"Model not found at {LOCAL_MODEL_PATH}. Please run download_models.py first.")
            return False
            
        # Try to use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        torch_dtype = torch.float16 if device == 0 else torch.float32
        
        logger.info(f"Loading model from {LOCAL_MODEL_PATH}...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_PATH))
        model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_MODEL_PATH),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        
        # Move model to GPU if available
        if device == 0:
            model = model.cuda()
        
        # Initialize the text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            do_sample=True,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
        )
        
        logger.info(f"Successfully loaded model: {MODEL_NAME} on {'GPU' if device == 0 else 'CPU'}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Load the model when the module is imported
load_local_model()

def generate_health_tips(patient_data, symptoms=None, language="en"):
    """
    Generate personalized health tips based on patient data and symptoms
    
    Args:
        patient_data (dict): Patient demographic information and medical history
        symptoms (list, optional): List of symptoms reported by the patient
        language (str): Language code for the response (e.g., 'en', 'sw', 'fr')
        
    Returns:
        dict: JSON response containing health tips and recommendations
    """
    # Prepare prompt based on patient data and symptoms
    patient_age = patient_data.get('age', 'unknown')
    patient_gender = patient_data.get('gender', 'unknown')
    patient_location = patient_data.get('location', 'unknown')
    
    prompt = f"""
    You are a medical assistant providing personalized health tips for a patient in Africa.
    
    Patient Information:
    - Age: {patient_age}
    - Gender: {patient_gender}
    - Location: {patient_location}
    """
    
    if symptoms:
        prompt += "\nReported Symptoms:\n"
        for symptom in symptoms:
            severity = symptom.get('severity', 'Unknown')
            category = symptom.get('category', 'General')
            prompt += f"- {symptom.get('description', 'Unknown symptom')} (Severity: {severity}, Category: {category})\n"
    
    prompt += """
    Based on the above information, provide 3-5 personalized health tips and recommendations.
    Consider:
    1. Age-appropriate advice
    2. Gender-specific health concerns if relevant
    3. Regional health issues common in their location
    4. Specific advice for managing reported symptoms
    5. Simple lifestyle recommendations
    
    Format your response as JSON with the following structure:
    {
        "health_tips": [
            {"title": "Brief title", "description": "Detailed explanation of the health tip"}
        ],
        "symptom_management": [
            {"symptom": "Specific symptom", "advice": "Advice for managing this symptom"}
        ],
        "follow_up": "Advice on when to seek further medical attention"
    }
    
    Keep all descriptions concise, clear, and culturally appropriate for the region.
    """
    
    # Define appropriate language models for different languages
    language_prompts = {
        "en": "Provide the response in English.",
        "sw": "Provide the response in Swahili.",
        "fr": "Provide the response in French.",
        "or": "Provide the response in Oromo if possible, otherwise in English.",
        "so": "Provide the response in Somali if possible, otherwise in English.",
        "am": "Provide the response in Amharic if possible, otherwise in English."
    }
    
    # Add language instruction
    lang_instruction = language_prompts.get(language, language_prompts["en"])
    prompt += f"\n\n{lang_instruction}"
    
    if not generator:
        logger.error("Hugging Face model not initialized")
        return {
            "health_tips": [{"title": "Error", "description": "AI service is not available."}],
            "follow_up": "Please try again later or contact support."
        }
    
    try:
        logger.info("Generating health tips with Hugging Face model")
        
        # Format the prompt for the model
        system_prompt = "You are a medical advisor with expertise in African healthcare contexts, providing culturally appropriate health advice. Always respond in valid JSON format."
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Generate response
        response = generator(
            full_prompt,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Extract the generated text
        generated_text = response[0]['generated_text']
        
        # Try to extract JSON from the response
        try:
            # Find the first { and last } to extract JSON
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}')
            if json_start >= 0 and json_end > json_start:
                json_str = generated_text[json_start:json_end+1]
                result = json.loads(json_str)
                logger.info("Successfully generated health tips with Hugging Face model")
                return result
            else:
                raise ValueError("No valid JSON found in the response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            logger.debug(f"Generated text: {generated_text}")
            return {
                "health_tips": [{"title": "Error", "description": "Error processing the response."}],
                "follow_up": "Please try again or contact support if the issue persists."
            }
            
    except Exception as e:
        logger.error(f"Error generating health tips: {str(e)}")
        return {
            "health_tips": [{"title": "Error", "description": "Unable to generate health tips at this time."}],
            "follow_up": "Please try again later or contact support."
        }

def generate_health_education(topic, language="en"):
    """
    Generate general health education content on a specific topic
    
    Args:
        topic (str): Health topic to generate information about
        language (str): Language code for the response
        
    Returns:
        dict: JSON response containing educational content
    """
    language_prompts = {
        "en": "Provide the response in English.",
        "sw": "Provide the response in Swahili.",
        "fr": "Provide the response in French.",
        "or": "Provide the response in Oromo if possible, otherwise in English.",
        "so": "Provide the response in Somali if possible, otherwise in English.",
        "am": "Provide the response in Amharic if possible, otherwise in English."
    }
    
    lang_instruction = language_prompts.get(language, language_prompts["en"])
    
    prompt = f"""
    Generate educational health content about "{topic}" for patients in rural Africa.
    
    Keep the information:
    1. Simple and easy to understand
    2. Relevant to typical health conditions in Africa
    3. Actionable with limited resources
    4. Culturally appropriate
    
    Format your response as JSON with the following structure:
    {{
        "title": "Title for the health topic",
        "overview": "Brief overview of the topic",
        "key_points": [
            "Important point 1",
            "Important point 2"
        ],
        "prevention": [
            "Prevention tip 1",
            "Prevention tip 2"
        ],
        "when_to_seek_help": "Guidance on when to seek medical assistance"
    }}
    
    {lang_instruction}
    """
    
    if not generator:
        logger.error("Hugging Face model not initialized")
        return {
            "title": topic,
            "overview": "AI service is not available.",
            "key_points": ["Please try again later."],
            "prevention": ["Contact support if the issue persists."],
            "when_to_seek_help": "If you have concerns, please contact a healthcare facility."
        }
    
    try:
        logger.info(f"Generating health education about {topic} with Hugging Face model")
        
        # Format the prompt for the model
        system_prompt = "You are a health educator specializing in public health in African communities. Always respond in valid JSON format."
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Generate response
        response = generator(
            full_prompt,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Extract the generated text
        generated_text = response[0]['generated_text']
        
        # Try to extract JSON from the response
        try:
            # Find the first { and last } to extract JSON
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}')
            if json_start >= 0 and json_end > json_start:
                json_str = generated_text[json_start:json_end+1]
                result = json.loads(json_str)
                logger.info(f"Successfully generated health education about {topic}")
                return result
            else:
                raise ValueError("No valid JSON found in the response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            logger.debug(f"Generated text: {generated_text}")
            return {
                "title": topic,
                "overview": "Error processing the response.",
                "key_points": ["Please try again."],
                "prevention": ["Contact support if the issue persists."],
                "when_to_seek_help": "If you have concerns, please contact a healthcare facility."
            }
            
    except Exception as e:
        logger.error(f"Error generating health education: {str(e)}")
        return {
            "title": topic,
            "overview": "Information temporarily unavailable.",
            "key_points": ["Please try again later."],
            "prevention": ["Contact support if the issue persists."],
            "when_to_seek_help": "If you have concerns, please contact a healthcare facility."
        }