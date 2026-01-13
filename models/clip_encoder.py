"""
Text Encoder Models for 4D Text Grounding

This module provides text encoding functionality using pretrained models,
primarily CLIP's text encoder for generating text embeddings from action
descriptions and prompts.
"""

import torch
import torch.nn as nn
try:
    import clip
except ImportError:
    clip = None
import logging
from typing import List, Union, Optional, Tuple
try:
    from transformers import CLIPTokenizer, CLIPTextModel
except ImportError:
    CLIPTokenizer, CLIPTextModel = None, None
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['CLIPTextEncoder', 'create_text_encoder']


class CLIPTextEncoder(nn.Module):
    """
    CLIP-based text encoder for action descriptions and prompts.
    
    Uses the pretrained CLIP text encoder to convert text descriptions
    into fixed-size embeddings suitable for multimodal learning tasks.
    """
    
    def __init__(
        self, 
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        freeze_encoder: bool = True,
        max_length: int = 77,
        add_projection: bool = False,
        projection_dim: int = 512
    ):
        """
        Initialize CLIP text encoder.
        
        Args:
            model_name: CLIP model variant to use
            device: Device to load model on
            freeze_encoder: Whether to freeze the pretrained weights
            max_length: Maximum token length for text processing
            add_projection: Whether to add a projection layer after CLIP
            projection_dim: Dimension of projection layer if used
        """
        super(CLIPTextEncoder, self).__init__()
        
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.freeze_encoder = freeze_encoder
        self.add_projection = add_projection
        
        # Load CLIP model
        if clip is None:
            raise ImportError(
                "CLIP is not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        
        try:
            self.clip_model, self.preprocess = clip.load(model_name, device=device)
            self.tokenizer = clip.tokenize
            
            # Ensure CLIP model is in float32 to avoid mixed precision issues
            self.clip_model = self.clip_model.float()
            
            logger.info(f"Loaded CLIP model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model {model_name}: {e}")
            raise
        
        # Get text encoder and embedding dimension
        self.text_encoder = self.clip_model
        # For CLIP, text_projection is a Parameter tensor with shape [text_dim, embed_dim]
        self.embedding_dim = self.clip_model.text_projection.shape[1]
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            logger.info("Frozen CLIP text encoder weights")
        
        # Optional projection layer
        self.projection = None
        if add_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.embedding_dim, projection_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_dim, projection_dim),
                nn.LayerNorm(projection_dim)
            )
            self.output_dim = projection_dim
            logger.info(f"Added projection layer: {self.embedding_dim} → {projection_dim}")
        else:
            self.output_dim = self.embedding_dim
        
        logger.info(f"Text encoder output dimension: {self.output_dim}")
    
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text string or list of text strings
        
        Returns:
            embeddings: [batch_size, output_dim] text embeddings
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        try:
            tokens = self.tokenizer(texts, truncate=True).to(self.device)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise
        
        # Encode with CLIP
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            text_features = self.text_encoder.encode_text(tokens)
            
            # Convert to float32 to avoid mixed precision issues
            text_features = text_features.float()
            
            # Normalize features (CLIP standard)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Apply projection if available
        if self.projection is not None:
            text_features = self.projection(text_features)
        
        return text_features
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Forward pass for text encoding.
        
        Args:
            texts: Input text(s) to encode
        
        Returns:
            embeddings: Text embeddings
        """
        return self.encode_text(texts)
    
    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.output_dim
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts for better encoding.
        
        Args:
            texts: List of raw text strings
        
        Returns:
            processed_texts: Cleaned and formatted texts
        """
        processed = []
        for text in texts:
            # Basic cleaning
            text = text.strip()
            
            # Add prompt prefix for better CLIP understanding
            if not text.lower().startswith(("a person", "someone", "the person")):
                text = f"A person is {text.lower()}"
            
            processed.append(text)
        
        return processed
    
    def encode_batch_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        preprocess: bool = True
    ) -> torch.Tensor:
        """
        Encode a large batch of texts efficiently.
        
        Args:
            texts: List of text strings to encode
            batch_size: Processing batch size
            preprocess: Whether to preprocess texts
        
        Returns:
            embeddings: [len(texts), output_dim] text embeddings
        """
        if preprocess:
            texts = self.preprocess_texts(texts)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode_text(batch_texts)
            all_embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)


class HuggingFaceCLIPTextEncoder(nn.Module):
    """
    Alternative CLIP text encoder using HuggingFace transformers.
    
    Provides more flexibility and control over the CLIP text encoder
    compared to the original OpenAI implementation.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        freeze_encoder: bool = True,
        max_length: int = 77
    ):
        """
        Initialize HuggingFace CLIP text encoder.
        
        Args:
            model_name: HuggingFace CLIP model name
            device: Device to load model on
            freeze_encoder: Whether to freeze pretrained weights
            max_length: Maximum sequence length
        """
        super(HuggingFaceCLIPTextEncoder, self).__init__()
        
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.freeze_encoder = freeze_encoder
        
        # Load tokenizer and model
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name).to(device)
            logger.info(f"Loaded HuggingFace CLIP model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace CLIP model: {e}")
            raise
        
        self.embedding_dim = self.text_model.config.hidden_size
        self.output_dim = self.embedding_dim
        
        # Freeze if requested
        if freeze_encoder:
            for param in self.text_model.parameters():
                param.requires_grad = False
            logger.info("Frozen HuggingFace CLIP text encoder weights")
    
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode texts using HuggingFace CLIP."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            outputs = self.text_model(**inputs)
            # Use pooled output (CLS token representation)
            embeddings = outputs.pooler_output
        
        return embeddings
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Forward pass."""
        return self.encode_text(texts)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.output_dim


def create_text_encoder(
    encoder_type: str = "clip",
    model_name: Optional[str] = None,
    device: str = "cuda",
    **kwargs
) -> Union[CLIPTextEncoder, HuggingFaceCLIPTextEncoder]:
    """
    Factory function to create text encoders.
    
    Args:
        encoder_type: Type of encoder ("clip" or "huggingface_clip")
        model_name: Specific model name (uses defaults if None)
        device: Device to load model on
        **kwargs: Additional arguments for encoder
    
    Returns:
        Text encoder instance
    
    Examples:
        # Create basic CLIP encoder
        encoder = create_text_encoder("clip")
        
        # Create with custom settings
        encoder = create_text_encoder(
            "clip", 
            model_name="ViT-L/14",
            freeze_encoder=False,
            add_projection=True
        )
    """
    if encoder_type == "clip":
        default_model = "ViT-B/32" if model_name is None else model_name
        return CLIPTextEncoder(model_name=default_model, device=device, **kwargs)
    
    elif encoder_type == "huggingface_clip":
        default_model = "openai/clip-vit-base-patch32" if model_name is None else model_name
        return HuggingFaceCLIPTextEncoder(model_name=default_model, device=device, **kwargs)
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_available_clip_models() -> List[str]:
    """Get list of available CLIP model variants."""
    return [
        "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
        "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
    ]


def estimate_text_encoding_time(
    texts: List[str], 
    encoder: Union[CLIPTextEncoder, HuggingFaceCLIPTextEncoder],
    num_runs: int = 10
) -> Tuple[float, float]:
    """
    Estimate text encoding time for performance analysis.
    
    Args:
        texts: Sample texts to encode
        encoder: Text encoder to test
        num_runs: Number of runs for averaging
    
    Returns:
        (avg_time_per_text, total_time) in seconds
    """
    import time
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = encoder.encode_text(texts)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_total_time = np.mean(times)
    avg_time_per_text = avg_total_time / len(texts)
    
    return avg_time_per_text, avg_total_time