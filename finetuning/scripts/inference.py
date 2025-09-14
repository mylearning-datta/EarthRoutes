#!/usr/bin/env python3
"""
Inference script for the trained travel sustainability model.
Loads the fine-tuned model and provides predictions for natural language queries.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

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
    print("Warning: transformers or peft not available. Install with:")
    print("pip install transformers peft torch")


class TravelSustainabilityModel:
    """Wrapper for the trained travel sustainability model."""
    
    def __init__(self, model_path: str):
        """Initialize the model from saved path."""
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            print("Cannot load model: Required packages not installed")
            return
        
        if not self.model_path.exists():
            print(f"Model path does not exist: {self.model_path}")
            return
        
        try:
            print(f"Loading model from {self.model_path}")
            
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
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
    
    def predict(self, query: str, max_new_tokens: int = 200) -> Dict:
        """Generate prediction for a natural language query."""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
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
                "query": query,
                "response": response,
                "full_output": full_response
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None


def test_model_inference(model_path: str = "finetuning/models/travel-sustainability-lora"):
    """Test the trained model with sample queries."""
    print("Testing Travel Sustainability Model")
    print("=" * 50)
    
    # Initialize model
    model = TravelSustainabilityModel(model_path)
    
    if not model.is_loaded():
        print("Model not loaded. Please train the model first:")
        print("python finetuning/scripts/train_sft.py")
        return
    
    # Test queries
    test_queries = [
        "I want to travel from Bangalore to Chennai and provide me with suitable sustainable options",
        "How should I go from Mumbai to Delhi in an eco-friendly way?",
        "I'm visiting Bangalore, suggest some sustainable places to visit",
        "What's the best green way to travel from Pune to Lonavala?",
        "Show me sustainable places in Mumbai"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        result = model.predict(query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Response: {result['response']}")
    
    print("\n" + "=" * 50)
    print("Inference test complete!")


def interactive_mode(model_path: str = "finetuning/models/travel-sustainability-lora"):
    """Interactive mode for testing queries."""
    print("Interactive Travel Sustainability Assistant")
    print("=" * 50)
    print("Type 'quit' to exit")
    
    # Initialize model
    model = TravelSustainabilityModel(model_path)
    
    if not model.is_loaded():
        print("Model not loaded. Please train the model first:")
        print("python finetuning/scripts/train_sft.py")
        return
    
    while True:
        try:
            query = input("\nEnter your travel query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("Generating response...")
            result = model.predict(query)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nResponse: {result['response']}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Travel Sustainability Model Inference")
    parser.add_argument("--model-path", default="finetuning/models/travel-sustainability-lora",
                       help="Path to the trained model")
    parser.add_argument("--mode", choices=["test", "interactive"], default="test",
                       help="Mode: test with sample queries or interactive")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        test_model_inference(args.model_path)
    else:
        interactive_mode(args.model_path)


if __name__ == "__main__":
    main()
