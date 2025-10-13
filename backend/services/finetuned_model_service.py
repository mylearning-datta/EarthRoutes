#!/usr/bin/env python3
"""
Service for loading and using the fine-tuned travel sustainability model.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any
import os
import logging
import re
import time

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

# MLX optional imports (for macOS 4-bit inference)
try:
    from mlx_lm import load as mlx_load, generate as mlx_generate  # type: ignore
    MLX_AVAILABLE = True
except Exception:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)

class FineTunedModelService:
    """Service for the trained travel sustainability model."""
    
    def __init__(self, model_path: str = None, variant: Optional[str] = None):
        """Initialize the model service."""
        # Lazy import to avoid circulars
        try:
            from config.settings import settings  # type: ignore
        except Exception:
            settings = None  # Fallback to env directly

        # Resolve model variant: only two supported values: 'community' | 'finetuned'
        # Default from env MODEL_MODE, else 'community'
        env_mode = os.getenv("MODEL_MODE", "community").strip().lower()
        if variant and variant.strip().lower() in {"community", "finetuned"}:
            self.model_variant = variant.strip().lower()
        elif env_mode in {"community", "finetuned"}:
            self.model_variant = env_mode
        else:
            self.model_variant = "community"

        # Configure MLX paths based on chosen variant
        # Community base (full MLX model dir)
        community_path = "/Users/arpita/Documents/project/finetuning/models/mistral-7b-instruct-4bit-mlx"
        # Finetuned adapters (LoRA adapter dir)
        finetuned_adapter_path = "/Users/arpita/Documents/project/finetuning/models/mistral-mlx-lora"

        os.environ["USE_MLX"] = "true"
        if self.model_variant == "finetuned":
            os.environ["MODEL_MODE"] = "finetuned"
            os.environ["MLX_MODEL"] = community_path
            os.environ["MLX_ADAPTER_PATH"] = finetuned_adapter_path
        else:
            os.environ["MODEL_MODE"] = "community"
            os.environ["MLX_MODEL"] = community_path
            # Clear adapter path for community mode
            os.environ.pop("MLX_ADAPTER_PATH", None)

        logger.info(
            f"Configured MLX for variant={self.model_variant} MLX_MODEL={os.environ.get('MLX_MODEL')} "
            f"MLX_ADAPTER_PATH={os.environ.get('MLX_ADAPTER_PATH','')}"
        )

        # For non-MLX fallbacks, still set model_path to community path
        self.model_path = self._select_and_prepare_model_path(settings, model_path)
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        # Initialize RAG services
        self.semantic_search = None
        self.co2_service = None
        self.maps_service = None
        t_start = time.perf_counter()
        self._init_rag_services()
        t_rag = time.perf_counter()
        self._load_model()
        t_loaded = time.perf_counter()
        logger.info(
            f"[timer] FineTunedModelService init: rag_init={t_rag - t_start:.3f}s, model_load={t_loaded - t_rag:.3f}s, total={t_loaded - t_start:.3f}s"
        )
        # Default backend flag if not set by loader
        if not hasattr(self, "using_mlx"):
            self.using_mlx = False

    def _resolve_path(self, raw_path: Optional[str]) -> Path:
        """Resolve a possibly relative path against project root."""
        if not raw_path:
            return Path("")
        p = Path(raw_path)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    def _select_and_prepare_model_path(self, settings, override_model_path: Optional[str]) -> Path:
        """Select model path based on env/settings and ensure availability (extract zip if needed)."""
        # For MLX we don't need a single model_path; return community path as base for non-MLX fallback
        return self._resolve_path("/Users/arpita/Documents/project/finetuning/models/mistral-7b-instruct-4bit-mlx")
    
    def _init_rag_services(self):
        """Initialize RAG services for enhanced context."""
        try:
            # Add backend to path for imports
            import sys
            from pathlib import Path
            backend_path = Path(__file__).parent.parent
            if str(backend_path) not in sys.path:
                sys.path.insert(0, str(backend_path))
            
            # Import services
            from services.semantic_search_service import semantic_search_service
            from services.co2_service import CO2EmissionService
            from services.google_maps import GoogleMapsService
            
            self.semantic_search = semantic_search_service
            self.co2_service = CO2EmissionService()
            self.maps_service = GoogleMapsService()
            logger.info("RAG services initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize RAG services: {e}")
            # Set to None to avoid errors
            self.semantic_search = None
            self.co2_service = None
            self.maps_service = None
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        t_load_start = time.perf_counter()
        use_mlx = os.getenv("USE_MLX", "false").lower() in {"1", "true", "yes"}
        if use_mlx:
            if not MLX_AVAILABLE:
                logger.error("USE_MLX is set but mlx packages are not available. Install mlx and mlx-lm.")
                return
            try:
                # Simple switch: community vs finetuned
                model_mode = os.getenv("MODEL_MODE", "community").lower()
                if model_mode == "finetuned":
                    # Load community base with local LoRA adapters
                    base_id = os.getenv("MLX_MODEL", "/Users/arpita/Documents/project/finetuning/models/mistral-7b-instruct-4bit-mlx")
                    adapter_path = os.getenv("MLX_ADAPTER_PATH", "/Users/arpita/Documents/project/finetuning/models/mistral-mlx-lora").strip()
                    if not adapter_path:
                        logger.error("MODEL_MODE=finetuned but MLX_ADAPTER_PATH is not set")
                        self.is_loaded = False
                        return
                    ap = Path(adapter_path)
                    if not ap.exists():
                        logger.error(f"Adapter path does not exist: {adapter_path}")
                        self.is_loaded = False
                        return
                    # Extra diagnostics about adapter files
                    try:
                        adapter_file = ap / "adapters.safetensors"
                        adapter_cfg = ap / "adapter_config.json"
                        if adapter_file.exists():
                            logger.info(f"MLX adapters file: {adapter_file} ({adapter_file.stat().st_size} bytes)")
                        if adapter_cfg.exists():
                            logger.info(f"MLX adapter config: {adapter_cfg} ({adapter_cfg.stat().st_size} bytes)")
                    except Exception as _e:
                        logger.debug(f"Adapter file introspection skipped: {_e}")

                    logger.info(f"Loading MLX base model '{base_id}' with adapters from '{adapter_path}'")
                    t0 = time.perf_counter()
                    self.model, self.tokenizer = mlx_load(base_id, adapter_path=adapter_path)
                    logger.info(f"[timer] MLX base+adapters load={time.perf_counter() - t0:.3f}s")
                    logger.info("Applied MLX LoRA adapters successfully")
                else:
                    mlx_model_id = os.getenv("MLX_MODEL", "/Users/arpita/Documents/project/finetuning/models/mistral-7b-instruct-4bit-mlx")
                    logger.info(f"Loading MLX community model: {mlx_model_id}")
                    t0 = time.perf_counter()
                    self.model, self.tokenizer = mlx_load(mlx_model_id)
                    logger.info(f"[timer] MLX community load={time.perf_counter() - t0:.3f}s")
                self.using_mlx = True
                self.is_loaded = True
                logger.info(f"[timer] MLX total_load={time.perf_counter() - t_load_start:.3f}s")
                return
            except Exception as e:
                logger.error(f"Failed to load MLX model: {e}")
                self.is_loaded = False
                return
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Cannot load model: transformers/peft not installed and USE_MLX is false")
            return
        
        if not self.model_path.exists():
            logger.error(f"Model path does not exist: {self.model_path}")
            return
        
        try:
            logger.info(f"Loading fine-tuned model from {self.model_path}")
            
            adapter_config_path = self.model_path / "adapter_config.json"

            # Case A: Directory is an adapter (has adapter_config.json) -> load base and apply LoRA
            if adapter_config_path.exists():
                # Load tokenizer from adapter dir
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Determine base model: env override > adapter_config > default gpt2
                base_model_name = os.getenv("FINETUNED_BASE_MODEL") or "gpt2"
                try:
                    import json as _json
                    with open(adapter_config_path, "r", encoding="utf-8") as f:
                        adapter_cfg = _json.load(f)
                        if not os.getenv("FINETUNED_BASE_MODEL"):
                            base_model_name = adapter_cfg.get("base_model_name_or_path", base_model_name)
                except Exception:
                    pass

                # Extra diagnostics about adapter files
                try:
                    adapter_file = self.model_path / "adapter_model.safetensors"
                    if not adapter_file.exists():
                        adapter_file = self.model_path / "adapters.safetensors"
                    if adapter_file.exists():
                        logger.info(f"Transformers adapters file: {adapter_file} ({adapter_file.stat().st_size} bytes)")
                    logger.info(f"Adapter config: {adapter_config_path} ({adapter_config_path.stat().st_size} bytes)")
                except Exception as _e:
                    logger.debug(f"Adapter file introspection skipped: {_e}")

                logger.info(f"Using base model for adapter: {base_model_name}")

                t0 = time.perf_counter()
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    load_in_8bit=True if torch.cuda.is_available() else False
                )
                logger.info(f"[timer] HF base_model_load={time.perf_counter() - t0:.3f}s")

                # Apply LoRA
                t1 = time.perf_counter()
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                self.model.eval()
                logger.info(f"[timer] HF peft_apply={time.perf_counter() - t1:.3f}s")
                logger.info("Applied Transformers LoRA adapters successfully")

            else:
                # Case B: Directory is a base model (placeholder) -> load directly without LoRA
                logger.info("No adapter_config.json found; treating model path as base model directory.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                t0 = time.perf_counter()
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    load_in_8bit=True if torch.cuda.is_available() else False
                )
                self.model.eval()
                logger.info(f"[timer] HF base_dir_model_load={time.perf_counter() - t0:.3f}s")
            
            self.is_loaded = True
            self.using_mlx = False
            logger.info(f"[timer] HF total_load={time.perf_counter() - t_load_start:.3f}s")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            # Fallback: if bitsandbytes or quantized weights cause issues, try MLX 4-bit
            try:
                msg = str(e).lower()
                if ("bitsandbytes" in msg or "bnb" in msg or "quant" in msg) and MLX_AVAILABLE:
                    mlx_model_id = os.getenv("MLX_MODEL", "mlx-community/Mistral-7B-Instruct-v0.2-4bit")
                    logger.warning("Falling back to MLX 4-bit model due to transformer load error")
                    self.model, self.tokenizer = mlx_load(mlx_model_id)
                    self.using_mlx = True
                    self.is_loaded = True
                    logger.info("MLX fallback model loaded successfully")
                    return
            except Exception as e2:
                logger.error(f"MLX fallback also failed: {e2}")
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
    
    def predict(self, query: str, max_new_tokens: int = 5000) -> Dict:
        """Generate prediction for a natural language query with RAG context."""
        t_predict_start = time.perf_counter()
        if not self.is_loaded:
            return {
                "success": False,
                "error": "Model not loaded. Please ensure the fine-tuned model is available.",
                "response": "I'm sorry, the fine-tuned model is not available at the moment. Please try the regular chat assistant."
            }
        
        try:
            # Extract travel context and gather RAG data
            t0 = time.perf_counter()
            travel_context = self._extract_travel_context(query)
            t1 = time.perf_counter()
            rag_data = self._gather_rag_data(query, travel_context)
            t2 = time.perf_counter()
            
            # Build enhanced prompt with RAG context
            enhanced_prompt = self._build_enhanced_prompt(query, rag_data, travel_context)
            t3 = time.perf_counter()
            
            use_mlx_env = os.getenv("USE_MLX", "false").lower() in {"1", "true", "yes"}
            use_mlx = (self.using_mlx or use_mlx_env) and MLX_AVAILABLE and self.model is not None and self.tokenizer is not None
            # Constrain generation length (looser cap for community mode)
            model_mode = os.getenv("MODEL_MODE", "community").lower()
            if model_mode == "community":
                safe_max_tokens = max(128, min(max_new_tokens, 2048))
            else:
                safe_max_tokens = max(64, min(max_new_tokens, 500))
            if use_mlx:
                # Free-form generation (kept short, with stops)
                prompt = enhanced_prompt
                # Community mode: fewer stop markers to avoid premature cut-off
                if model_mode == "community":
                    stop_markers = ["```"]
                else:
                    stop_markers = ["\n###", "\nRESPONSE_JSON:", "```", "\nRecommendations", "\nSources"]
                try:
                    t_gen_start = time.perf_counter()
                    text = mlx_generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        max_tokens=safe_max_tokens,
                        stop=stop_markers,
                        verbose=False
                    )
                    t_gen_end = time.perf_counter()
                except TypeError:
                    t_gen_start = time.perf_counter()
                    text = mlx_generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        max_tokens=safe_max_tokens,
                        verbose=False
                    )
                    t_gen_end = time.perf_counter()
                raw_response = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
                response = self._sanitize_response_text(raw_response)
                # Log the raw MLX output for observability
                try:
                    logger.info(f"[mlx_output] {response}")
                except Exception:
                    pass
                logger.info(
                    f"[timer] predict(MLX): extract={t1 - t0:.3f}s, gather_rag={t2 - t1:.3f}s, build_prompt={t3 - t2:.3f}s, generate={t_gen_end - t_gen_start:.3f}s, total={time.perf_counter() - t_predict_start:.3f}s"
                )
                return {
                    "success": True,
                    "response": response,
                    "query": query,
                    "model_type": "mlx",
                    "rag_context": rag_data
                }
            else:
                # transformers path
                prompt = enhanced_prompt
                t_tok0 = time.perf_counter()
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024  # Increased for RAG context
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    t_gen_start = time.perf_counter()
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=safe_max_tokens,
                        do_sample=False,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                    t_gen_end = time.perf_counter()
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                raw_response = full_response[len(prompt):].strip()
                response = self._sanitize_response_text(raw_response)
                logger.info(
                    f"[timer] predict(HF): extract={t1 - t0:.3f}s, gather_rag={t2 - t1:.3f}s, build_prompt={t3 - t2:.3f}s, tokenize={t_tok0 - t3:.3f}s, generate={t_gen_end - t_gen_start:.3f}s, total={time.perf_counter() - t_predict_start:.3f}s"
                )
                return {
                    "success": True,
                    "response": response,
                    "query": query,
                    "model_type": "fine-tuned",
                    "rag_context": rag_data
                }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "error": f"Prediction failed: {e}",
                "response": "I'm sorry, I encountered an error while generating a response. Please try again."
            }

    
    
    def _extract_travel_context(self, query: str) -> Dict[str, Any]:
        """Extract travel-related context from user query."""
        context = {
            "cities": [],
            "travel_modes": [],
            "intent": "general",
            "keywords": [],
            "has_distance_query": False,
            "has_place_query": False,
            "has_hotel_query": False
        }
        
        query_lower = query.lower()
        
        # Extract cities from database
        try:
            from utils.postgres_database import postgres_db_manager
            available_cities = postgres_db_manager.get_cities()
            for city in available_cities:
                if city.lower() in query_lower:
                    context["cities"].append(city)
        except Exception as e:
            logger.warning(f"Could not get cities from database: {e}")
            # Fallback to common cities
            fallback_cities = ["delhi", "mumbai", "bangalore", "hyderabad", "kolkata", "chennai", "pune"]
            for city in fallback_cities:
                if city in query_lower:
                    context["cities"].append(city.title())
        
        # Extract travel modes from database
        try:
            from utils.postgres_database import postgres_db_manager
            available_modes = postgres_db_manager.get_travel_modes()
            for mode_id, mode_data in available_modes.items():
                mode_name = mode_data.get("name", "").lower()
                # Check for exact match or partial match
                if mode_name in query_lower or mode_id in query_lower:
                    context["travel_modes"].append(mode_id)
                else:
                    # Check for partial word matches
                    mode_words = mode_name.split()
                    for word in mode_words:
                        if word in query_lower and len(word) > 2:  # Avoid matching short words
                            context["travel_modes"].append(mode_id)
                            break
        except Exception as e:
            logger.warning(f"Could not get travel modes from database: {e}")
            # Fallback to common modes
            fallback_modes = ["flight", "train", "bus", "car", "bike", "bicycle", "walking"]
            for mode in fallback_modes:
                if mode in query_lower:
                    context["travel_modes"].append(mode)
        
        # Determine intent
        if any(word in query_lower for word in ["travel", "go", "visit", "from", "to"]):
            context["intent"] = "travel_planning"
            context["has_distance_query"] = True
        elif any(word in query_lower for word in ["place", "attraction", "monument", "temple", "park"]):
            context["intent"] = "place_search"
            context["has_place_query"] = True
        elif any(word in query_lower for word in ["hotel", "accommodation", "stay", "room"]):
            context["intent"] = "hotel_search"
            context["has_hotel_query"] = True
        elif any(word in query_lower for word in ["sustainable", "eco", "green", "carbon", "co2"]):
            context["intent"] = "sustainability"
        
        # Extract keywords
        context["keywords"] = query.split()
        
        return context
    
    def _gather_rag_data(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant data using RAG services."""
        rag_data = {
            "places": [],
            "hotels": [],
            "distance_info": None,
            "co2_info": None,
            "search_results": {}
        }
        
        try:
            # Search for places if relevant
            if context["has_place_query"] or context["intent"] in ["place_search", "travel_planning"]:
                if context["cities"]:
                    for city in context["cities"]:
                        places_result = self.semantic_search.search_places_natural_language(
                            query, city, limit=5
                        )
                        rag_data["places"].extend(places_result.get("results", []))
                else:
                    places_result = self.semantic_search.search_places_natural_language(
                        query, None, limit=5
                    )
                    rag_data["places"] = places_result.get("results", [])
            
            # Search for hotels if relevant
            if context["has_hotel_query"] or context["intent"] in ["hotel_search", "travel_planning"]:
                if context["cities"]:
                    for city in context["cities"]:
                        hotels_result = self.semantic_search.search_hotels_natural_language(
                            query, city, limit=5
                        )
                        rag_data["hotels"].extend(hotels_result.get("results", []))
                else:
                    hotels_result = self.semantic_search.search_hotels_natural_language(
                        query, None, limit=5
                    )
                    rag_data["hotels"] = hotels_result.get("results", [])
            
            # Calculate distance and CO2 if travel planning
            if context["has_distance_query"] and len(context["cities"]) >= 2:
                source = context["cities"][0]
                destination = context["cities"][1]
                
                # Get distance
                distance_info = self.maps_service.get_distance(source, destination)
                rag_data["distance_info"] = distance_info
                
                # Use the same method as ChatGPT path: delegate to travel_tools for mode list and emissions
                try:
                    from tools.travel_tools import travel_tools
                    comparison = travel_tools.compare_travel_modes(source, destination)
                    distance_km = distance_info["distance"]["value"] / 1000
                    options = comparison.get("options", [])
                    # Normalize to existing comparison structure used by prompt builder
                    co2_comparison = [
                        {
                            "distanceKm": distance_km,
                            "travelMode": opt.get("id", ""),
                            "travelModeName": opt.get("name", opt.get("id", "")),
                            "emissionFactor": float(opt.get("emissionFactor", 0.0) or 0.0),
                            "totalEmissions": float(opt.get("co2Emissions", 0.0) or 0.0),
                            "equivalentMetrics": {"treesNeeded": int(opt.get("treesNeeded", 0) or 0)}
                        }
                        for opt in options
                    ]
                    rag_data["co2_info"] = {
                        "distance_km": distance_km,
                        "comparison": co2_comparison
                    }
                except Exception as _e:
                    logger.warning(f"travel_tools comparison failed, skipping CO2 info: {_e}")
        
        except Exception as e:
            logger.warning(f"Error gathering RAG data: {e}")
        
        return rag_data
    
    def _build_enhanced_prompt(self, query: str, rag_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build ChatGPT-like prompt with RAG context."""
        
        # System message
        system_msg = """You are a helpful travel and sustainability assistant. You provide personalized, accurate, and environmentally conscious travel advice. 

IMPORTANT: Always include ALL the provided context details in your response:
- If distance information is provided, mention the exact distance and travel time
- If CO2 emissions data is provided, show the comparison for different travel modes with specific emission values
- If places are suggested, include their names, locations, and descriptions
- If hotels are suggested, include their names, ratings, and amenities
- Always provide specific, actionable recommendations with concrete details

Formatting requirement: Whenever you present options, lists, alternatives, itineraries, recommendations, or comparisons, format them as bulleted points.

Use the provided context to give comprehensive, detailed responses that include all relevant information."""
        
        # Build context section
        context_sections = []
        
        # Add places context (more concise)
        if rag_data["places"]:
            places_text = "\n".join([
                f"• {place.get('name', 'Unknown')} ({place.get('city', 'Unknown')}) - Rating: {place.get('google_review_rating', 'N/A')} - {place.get('type', 'N/A')} - Sustainable: {'Yes' if place.get('is_sustainable') else 'No'}"
                for place in rag_data["places"][:3]  # Show fewer places but with key info
            ])
            context_sections.append(f"PLACES: {places_text}")
        
        # Add hotels context (labeled for clarity)
        if rag_data["hotels"]:
            hotels_text = "\n".join([
                (
                    "• Name: " + str(hotel.get("name", "Unknown")) +
                    " | City: " + str(hotel.get("city", "Unknown")) +
                    " | Rating: " + str(hotel.get("rating", "N/A")) +
                    (" | Price: " + str(hotel.get("price_range")) if hotel.get("price_range") else "") +
                    (" | Amenities: " + str(hotel.get("amenities")) if hotel.get("amenities") else "")
                )
                for hotel in rag_data["hotels"][:3]
            ])
            context_sections.append(f"HOTELS: {hotels_text}")
        
        # Add distance and CO2 context (more concise)
        if rag_data["distance_info"] and rag_data["co2_info"]:
            distance = rag_data["distance_info"]
            co2 = rag_data["co2_info"]
            
            distance_text = f"DISTANCE: {distance['distance']['text']} ({distance['origin']} to {distance['destination']})"
            
            co2_text = "CO2 EMISSIONS: " + ", ".join([
                f"{mode.get('travelModeName') or mode['travelMode']}: {mode['totalEmissions']:.1f}kg"
                for mode in co2["comparison"]
            ])
            
            context_sections.append(f"{distance_text} | {co2_text}")
        
        # Combine context
        context_text = "\n\n".join(context_sections) if context_sections else "No specific travel context available."
        
        # Build final prompt
        prompt = f"""{system_msg}

Context:
{context_text}

User Query: {query}

IMPORTANT INSTRUCTIONS FOR YOUR RESPONSE:
1. Start your response by mentioning the travel distance and CO2 emissions if provided
2. Include specific place names, ratings, and descriptions from the context
3. Include specific hotel names, ratings, and amenities from the context
4. Always mention the exact CO2 values for different travel modes
5. Provide actionable recommendations with concrete details
6. Present any options, alternatives, or recommendations as bulleted points

Response:
"""
        
        return prompt

    def _sanitize_response_text(self, text: str) -> str:
        """Remove noisy heading markers like '###' / '######' and trim empty hash-only lines."""
        try:
            lines = text.splitlines()
            cleaned_lines = []
            for line in lines:
                s = line.strip()
                # Drop lines that are only hashes or empty
                if not s or set(s) <= {"#"}:
                    continue
                # Remove leading/trailing markdown-style headers (any count of '#')
                s = re.sub(r"^\s*#{1,6}\s*", "", s)
                s = re.sub(r"\s*#{1,6}\s*$", "", s)
                s = s.strip()
                if s:
                    cleaned_lines.append(s)
            # Collapse consecutive empty lines (already filtered) and rejoin
            return "\n".join(cleaned_lines)
        except Exception:
            return text

    def _format_structured_travel_response(self, rag_data: Dict[str, Any], best_mode: str, reason: str) -> str:
        distance = rag_data.get("distance_info", {}).get("distance", {}).get("text", "N/A")
        origin = rag_data.get("distance_info", {}).get("origin", "Unknown")
        destination = rag_data.get("distance_info", {}).get("destination", "Unknown")
        comparison = rag_data.get("co2_info", {}).get("comparison", [])
        bullets = []
        for m in comparison[:4]:
            bullets.append(f"- {m['travelMode']}: {m['totalEmissions']:.1f} kg CO2e")
        bullets_text = "\n".join(bullets)
        return (
            f"Distance: {distance} ({origin} → {destination})\n\n"
            f"CO2 by mode:\n{bullets_text}\n\n"
            f"Recommendation: {best_mode}\n- {reason.strip()}"
        )
    
    def get_model_status(self) -> Dict:
        """Get the current status of the model."""
        return {
            "is_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "device": "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else "unknown",
            "selected_variant": getattr(self, "model_variant", "unknown"),
            "using_mlx": getattr(self, "using_mlx", False),
            "backend": "mlx" if getattr(self, "using_mlx", False) else "transformers"
        }

# Global instances by variant
_finetuned_model_services: Dict[str, FineTunedModelService] = {}

def get_finetuned_model_service(variant: Optional[str] = None) -> FineTunedModelService:
    """Get a fine-tuned model service instance for the given variant (cached).
    Supported variants: 'community' | 'finetuned'. If None, falls back to env MODEL_MODE.
    """
    # Resolve to supported set only
    v_env = os.getenv("MODEL_MODE", "community").strip().lower()
    v = (variant or v_env)
    v = v.strip().lower() if v else "community"
    if v not in {"community", "finetuned"}:
        v = "community"

    if v not in _finetuned_model_services:
        _finetuned_model_services[v] = FineTunedModelService(variant=v)
    return _finetuned_model_services[v]
