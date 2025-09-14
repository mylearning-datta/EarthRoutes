#!/usr/bin/env python3
"""
Service for loading and using the fine-tuned travel sustainability model.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers or peft not available. Install with: pip install transformers peft torch")

logger = logging.getLogger(__name__)

class FineTunedModelService:
    """Service for the trained travel sustainability model."""
    
    def __init__(self, model_path: str = "../finetuning/models/travel-sustainability-lora"):
        """Initialize the model service."""
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Cannot load model: Required packages not installed")
            return
        
        if not self.model_path.exists():
            logger.error(f"Model path does not exist: {self.model_path}")
            return
        
        try:
            logger.info(f"Loading fine-tuned model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                load_in_8bit=True if torch.cuda.is_available() else False
            )
            
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            self.is_loaded = True
            logger.info("Fine-tuned model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
    
    def predict(self, query: str, max_new_tokens: int = 200) -> Dict:
        """Generate prediction for a natural language query."""
        if not self.is_loaded:
            return {
                "success": False,
                "error": "Model not loaded. Please ensure the fine-tuned model is available.",
                "response": "I'm sorry, the fine-tuned model is not available at the moment. Please try the regular chat assistant."
            }
        
        try:
            # Format the input
            prompt = f"### INSTRUCTION:\n{query}\n### RESPONSE:\n"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            return {
                "success": True,
                "response": response,
                "query": query,
                "model_type": "fine-tuned"
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "error": f"Prediction failed: {e}",
                "response": "I'm sorry, I encountered an error while generating a response. Please try again."
            }
    
    def get_model_status(self) -> Dict:
        """Get the current status of the model."""
        return {
            "is_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "device": "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else "unknown"
        }

# Global instance
_finetuned_model_service = None

def get_finetuned_model_service() -> FineTunedModelService:
    """Get the global fine-tuned model service instance."""
    global _finetuned_model_service
    if _finetuned_model_service is None:
        _finetuned_model_service = FineTunedModelService()
    return _finetuned_model_service
