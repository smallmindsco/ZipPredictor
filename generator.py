"""
Inference Module for Zip Predictor Model

This module handles:
- Loading trained models
- Generating zip files from text prompts
- Post-processing and validation of generated outputs
"""

import torch
import torch.nn.functional as F
import safetensors.torch as st
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import io
import zipfile
import json
import struct
from dataclasses import dataclass
import logging

from ..models.architecture import ZipPredictorModel, create_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for zip generation."""
    max_length: Optional[int] = None
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    num_beams: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    
    # Post-processing
    validate_zip: bool = True
    repair_zip: bool = True
    compression_level: int = 6


class ZipGenerator:
    """
    Generator for creating zip files from text prompts.
    
    This class wraps the trained model K and provides a high-level
    interface for generating valid zip files.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer: Any = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize generator.
        
        Args:
            model_path: Path to trained model checkpoint
            tokenizer: Tokenizer for encoding prompts
            device: Device to run inference on
            config: Generation configuration
        """
        self.device = device
        self.config = config or GenerationConfig()
        self.tokenizer = tokenizer
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"ZipGenerator initialized on {device}")
    
    def _load_model(self, model_path: Union[str, Path]) -> ZipPredictorModel:
        """Load model from checkpoint."""
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            model_config = {}
        
        # Create model
        model = ZipPredictorModel(**model_config)
        
        # Load weights
        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            state_dict = st.load_file(str(weights_path))
        else:
            state_dict = torch.load(model_path / "model.pt")
        
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_path}")
        
        return model
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> bytes:
        """
        Generate a zip file from a text prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Override generation config
            
        Returns:
            Zip file bytes
        """
        # Merge config with overrides
        gen_config = GenerationConfig(
            **{**self.config.__dict__, **kwargs}
        )
        
        # Tokenize prompt
        input_ids, attention_mask = self._encode_prompt(prompt)
        
        # Generate raw bytes
        raw_bytes = self.model.generate_zip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=gen_config.max_length,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p
        )[0]
        
        # Post-process
        if gen_config.validate_zip:
            if not self._is_valid_zip(raw_bytes):
                if gen_config.repair_zip:
                    raw_bytes = self._repair_zip(raw_bytes, gen_config)
                else:
                    logger.warning("Generated invalid zip file")
        
        return raw_bytes
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[bytes]:
        """
        Generate zip files for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Override generation config
            
        Returns:
            List of zip file bytes
        """
        # Merge config
        gen_config = GenerationConfig(
            **{**self.config.__dict__, **kwargs}
        )
        
        # Tokenize all prompts
        input_ids_list = []
        attention_mask_list = []
        max_len = 0
        
        for prompt in prompts:
            ids, mask = self._encode_prompt(prompt)
            input_ids_list.append(ids.squeeze(0))
            attention_mask_list.append(mask.squeeze(0))
            max_len = max(max_len, ids.size(1))
        
        # Pad to same length
        input_ids = torch.zeros(len(prompts), max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(len(prompts), max_len, dtype=torch.long, device=self.device)
        
        for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(mask)] = mask
        
        # Generate
        raw_bytes_list = self.model.generate_zip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=gen_config.max_length,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p
        )
        
        # Post-process each
        results = []
        for raw_bytes in raw_bytes_list:
            if gen_config.validate_zip and not self._is_valid_zip(raw_bytes):
                if gen_config.repair_zip:
                    raw_bytes = self._repair_zip(raw_bytes, gen_config)
            results.append(raw_bytes)
        
        return results
    
    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt to token IDs."""
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
        else:
            # Fallback: character-level
            input_ids = torch.tensor(
                [[ord(c) for c in prompt[:512]]],
                dtype=torch.long,
                device=self.device
            )
            attention_mask = torch.ones_like(input_ids)
        
        return input_ids, attention_mask
    
    def _is_valid_zip(self, data: bytes) -> bool:
        """Check if bytes form a valid zip file."""
        try:
            with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
                # Check for corruption
                return zf.testzip() is None
        except:
            return False
    
    def _repair_zip(self, data: bytes, config: GenerationConfig) -> bytes:
        """Attempt to repair malformed zip data."""
        # Strategy 1: Find valid zip header and extract
        zip_magic = b'PK\x03\x04'
        start_idx = data.find(zip_magic)
        
        if start_idx != -1:
            candidate = data[start_idx:]
            if self._is_valid_zip(candidate):
                return candidate
        
        # Strategy 2: Wrap raw content in a new zip
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=config.compression_level) as zf:
            zf.writestr('recovered_content.bin', data)
        
        return buffer.getvalue()
    
    def save_output(
        self,
        zip_bytes: bytes,
        output_path: Union[str, Path],
        extract: bool = False
    ):
        """
        Save generated zip to file.
        
        Args:
            zip_bytes: Generated zip bytes
            output_path: Path to save to
            extract: Whether to extract contents
        """
        output_path = Path(output_path)
        
        if extract:
            # Extract to directory
            output_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
                zf.extractall(output_path)
            logger.info(f"Extracted zip to {output_path}")
        else:
            # Save as zip file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(zip_bytes)
            logger.info(f"Saved zip to {output_path}")


class StreamingZipGenerator(ZipGenerator):
    """
    Generator with streaming support for very large outputs.
    
    Uses chunked generation to handle outputs larger than memory.
    """
    
    def __init__(self, *args, chunk_size: int = 8192, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size
    
    @torch.no_grad()
    def generate_streaming(
        self,
        prompt: str,
        output_path: Union[str, Path],
        **kwargs
    ):
        """
        Generate zip file with streaming output.
        
        Args:
            prompt: Input prompt
            output_path: Path to write output
            **kwargs: Generation config overrides
        """
        gen_config = GenerationConfig(
            **{**self.config.__dict__, **kwargs}
        )
        
        # Encode prompt
        input_ids, attention_mask = self._encode_prompt(prompt)
        
        # Encode
        encoder_output, encoder_pooled = self.model.text_encoder(
            input_ids, attention_mask
        )
        
        # Predict length
        predicted_length = self.model.content_decoder.length_predictor(encoder_pooled)
        max_length = int(predicted_length.max().item()) + 100
        
        # Open output file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            # Generate in chunks
            generated = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            chunk_buffer = []
            
            for _ in range(max_length):
                # Generate next token
                tgt_emb = self.model.content_decoder.output_embedding(generated)
                tgt_emb = self.model.content_decoder.pos_encoder(
                    tgt_emb.transpose(0, 1)
                ).transpose(0, 1)
                
                tgt_len = generated.size(1)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    tgt_len
                ).to(self.device)
                
                decoded = self.model.content_decoder.transformer_decoder(
                    tgt_emb, encoder_output, tgt_mask=tgt_mask
                )
                
                logits = self.model.content_decoder.output_projection(decoded[:, -1, :])
                logits = logits / gen_config.temperature
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if next_token.item() == 1:
                    break
                
                # Add to buffer
                chunk_buffer.append(next_token.item())
                
                # Flush chunk if needed
                if len(chunk_buffer) >= self.chunk_size:
                    f.write(bytes(chunk_buffer))
                    chunk_buffer = []
                
                # Update generated sequence
                generated = torch.cat([generated, next_token], dim=1)
            
            # Flush remaining
            if chunk_buffer:
                f.write(bytes(chunk_buffer))
        
        logger.info(f"Streamed output to {output_path}")


class ZipPredictorPipeline:
    """
    High-level pipeline for using the zip predictor.
    
    Provides a simple interface similar to HuggingFace pipelines.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer: Any = None,
        device: str = 'auto'
    ):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.generator = ZipGenerator(
            model_path=model_path,
            tokenizer=tokenizer,
            device=device
        )
    
    def __call__(
        self,
        prompts: Union[str, List[str]],
        output_dir: Optional[Union[str, Path]] = None,
        extract: bool = False,
        **kwargs
    ) -> Union[bytes, List[bytes]]:
        """
        Generate zip files from prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            output_dir: Optional directory to save outputs
            extract: Whether to extract zip contents
            **kwargs: Generation config overrides
            
        Returns:
            Generated zip bytes (single or list)
        """
        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]
        
        # Generate
        results = self.generator.generate_batch(prompts, **kwargs)
        
        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (prompt, zip_bytes) in enumerate(zip(prompts, results)):
                # Create filename from prompt
                safe_name = "".join(c for c in prompt[:50] if c.isalnum() or c in ' -_').strip()
                safe_name = safe_name.replace(' ', '_') or f"output_{i}"
                
                if extract:
                    self.generator.save_output(
                        zip_bytes,
                        output_dir / safe_name,
                        extract=True
                    )
                else:
                    self.generator.save_output(
                        zip_bytes,
                        output_dir / f"{safe_name}.zip",
                        extract=False
                    )
        
        if single_input:
            return results[0]
        return results


# Convenience function
def load_generator(
    model_path: Union[str, Path],
    **kwargs
) -> ZipPredictorPipeline:
    """
    Load a trained zip predictor model.
    
    Args:
        model_path: Path to model checkpoint
        **kwargs: Additional arguments for pipeline
        
    Returns:
        ZipPredictorPipeline instance
    """
    return ZipPredictorPipeline(model_path, **kwargs)
