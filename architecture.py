"""
Zip File Predictor Model Architecture

This module defines model K - a neural network that learns to predict
the zip file output of another model M given a text prompt.

Architecture Overview:
- Text Encoder: Encodes input prompts into latent space
- Content Predictor: Predicts the structured content that will be zipped
- Binary Encoder: Converts predicted content to binary-ready format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """
    Encodes text prompts into a rich latent representation.
    
    Uses a transformer encoder with learned embeddings to capture
    semantic meaning from input prompts.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        dim_feedforward: int = 3072,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Pooling projection
        self.pool_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) mask for padding
            
        Returns:
            sequence_output: (batch, seq_len, d_model) per-token representations
            pooled_output: (batch, d_model) sentence-level representation
        """
        # Embed tokens
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True = ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Encode
        sequence_output = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Pool: mean over non-padded tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled_output = (sequence_output * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        else:
            pooled_output = sequence_output.mean(dim=1)
        
        pooled_output = self.pool_projection(pooled_output)
        
        return sequence_output, pooled_output


class ContentDecoder(nn.Module):
    """
    Decodes latent representation into structured content.
    
    This module predicts the actual content that will be packed into
    the zip file, handling variable-length outputs through autoregressive
    or parallel decoding.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        output_vocab_size: int = 65536,  # Extended for byte-level output
        nhead: int = 12,
        num_layers: int = 8,
        dim_feedforward: int = 3072,
        max_output_len: int = 65536,  # Max bytes to predict
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_output_len = max_output_len
        self.output_vocab_size = output_vocab_size
        
        # Output embedding (for autoregressive decoding)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_output_len, dropout)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_vocab_size)
        
        # Length predictor
        self.length_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Ensure positive length
        )
        
    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_pooled: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_output: (batch, src_len, d_model) from text encoder
            encoder_pooled: (batch, d_model) pooled encoder output
            target_ids: (batch, tgt_len) target byte sequence for training
            target_mask: (batch, tgt_len) mask for padding
            
        Returns:
            Dictionary containing logits and predicted length
        """
        batch_size = encoder_output.size(0)
        
        # Predict output length
        predicted_length = self.length_predictor(encoder_pooled).squeeze(-1)
        
        if target_ids is not None:
            # Training mode: teacher forcing
            tgt_len = target_ids.size(1)
            
            # Embed targets
            tgt_emb = self.output_embedding(target_ids) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
            # Causal mask for autoregressive decoding
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_emb.device)
            
            # Decode
            decoded = self.transformer_decoder(
                tgt_emb,
                encoder_output,
                tgt_mask=tgt_mask
            )
            
            # Project to vocabulary
            logits = self.output_projection(decoded)
            
            return {
                'logits': logits,
                'predicted_length': predicted_length
            }
        else:
            # Inference mode: autoregressive generation
            return {
                'encoder_output': encoder_output,
                'predicted_length': predicted_length
            }
    
    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_pooled: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressively generate output bytes.
        
        Args:
            encoder_output: (batch, src_len, d_model)
            encoder_pooled: (batch, d_model)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            generated: (batch, gen_len) generated byte indices
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Predict length if not specified
        if max_length is None:
            predicted_len = self.length_predictor(encoder_pooled).squeeze(-1)
            max_length = int(predicted_len.max().item()) + 100  # Add buffer
            max_length = min(max_length, self.max_output_len)
        
        # Start with BOS token (0)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Embed current sequence
            tgt_emb = self.output_embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
            
            # Causal mask
            tgt_len = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
            
            # Decode
            decoded = self.transformer_decoder(
                tgt_emb,
                encoder_output,
                tgt_mask=tgt_mask
            )
            
            # Get logits for last position
            logits = self.output_projection(decoded[:, -1, :]) / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS (token 1)
            if (next_token == 1).all():
                break
        
        return generated


class FileStructurePredictor(nn.Module):
    """
    Predicts the structure of files within the zip archive.
    
    This module handles multi-file outputs by predicting:
    - Number of files
    - File names/paths
    - File types
    - Content distribution across files
    """
    
    def __init__(
        self,
        d_model: int = 768,
        max_files: int = 100,
        max_filename_len: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_files = max_files
        self.max_filename_len = max_filename_len
        
        # File count predictor
        self.file_count_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, max_files)
        )
        
        # File structure decoder
        self.file_query = nn.Parameter(torch.randn(max_files, d_model))
        
        file_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.file_decoder = nn.TransformerDecoder(file_decoder_layer, num_layers)
        
        # File metadata heads
        self.filename_head = nn.Linear(d_model, max_filename_len * 256)  # Char-level
        self.filetype_head = nn.Linear(d_model, 32)  # Common file types
        self.filesize_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )
        
    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_pooled: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict file structure from encoded prompt.
        
        Returns:
            Dictionary with file count logits, filenames, types, and sizes
        """
        batch_size = encoder_output.size(0)
        
        # Predict number of files
        file_count_logits = self.file_count_predictor(encoder_pooled)
        
        # Expand file queries for batch
        file_queries = self.file_query.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Decode file structure
        file_features = self.file_decoder(file_queries, encoder_output)
        
        # Predict metadata for each file slot
        filename_logits = self.filename_head(file_features).view(
            batch_size, self.max_files, self.max_filename_len, 256
        )
        filetype_logits = self.filetype_head(file_features)
        filesize_pred = self.filesize_head(file_features).squeeze(-1)
        
        return {
            'file_count_logits': file_count_logits,
            'filename_logits': filename_logits,
            'filetype_logits': filetype_logits,
            'filesize_predictions': filesize_pred,
            'file_features': file_features
        }


class ZipPredictorModel(nn.Module):
    """
    Complete model K that learns to predict zip file outputs.
    
    Architecture:
    1. TextEncoder: Encodes input prompt
    2. FileStructurePredictor: Predicts zip archive structure
    3. ContentDecoder: Generates content for each file
    4. ZipAssembler: Combines predictions into final zip bytes
    
    The model can be trained via:
    - Direct supervision: Given (prompt, zip_bytes) pairs
    - Distillation: Learning from model M's outputs
    - Hybrid: Combination of both approaches
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        encoder_layers: int = 6,
        decoder_layers: int = 8,
        nhead: int = 12,
        dim_feedforward: int = 3072,
        max_seq_len: int = 512,
        max_output_len: int = 65536,
        max_files: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'encoder_layers': encoder_layers,
            'decoder_layers': decoder_layers,
            'nhead': nhead,
            'dim_feedforward': dim_feedforward,
            'max_seq_len': max_seq_len,
            'max_output_len': max_output_len,
            'max_files': max_files,
            'dropout': dropout
        }
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # File structure predictor
        self.file_predictor = FileStructurePredictor(
            d_model=d_model,
            max_files=max_files,
            nhead=nhead,
            num_layers=4,
            dropout=dropout
        )
        
        # Content decoder
        self.content_decoder = ContentDecoder(
            d_model=d_model,
            output_vocab_size=65536,
            nhead=nhead,
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            max_output_len=max_output_len,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_bytes: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        file_counts: Optional[torch.Tensor] = None,
        filenames: Optional[torch.Tensor] = None,
        filetypes: Optional[torch.Tensor] = None,
        filesizes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_ids: (batch, seq_len) tokenized prompts
            attention_mask: (batch, seq_len) attention mask
            target_bytes: (batch, output_len) target zip bytes
            target_mask: (batch, output_len) mask for target
            file_counts: (batch,) number of files in each zip
            filenames: (batch, max_files, filename_len) encoded filenames
            filetypes: (batch, max_files) file type indices
            filesizes: (batch, max_files) file sizes
            
        Returns:
            Dictionary of model outputs and predictions
        """
        # Encode prompt
        encoder_output, encoder_pooled = self.text_encoder(input_ids, attention_mask)
        
        # Predict file structure
        file_structure = self.file_predictor(encoder_output, encoder_pooled)
        
        # Decode content
        content_output = self.content_decoder(
            encoder_output,
            encoder_pooled,
            target_ids=target_bytes,
            target_mask=target_mask
        )
        
        return {
            'encoder_output': encoder_output,
            'encoder_pooled': encoder_pooled,
            **file_structure,
            **content_output
        }
    
    @torch.no_grad()
    def generate_zip(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> List[bytes]:
        """
        Generate zip file bytes from a text prompt.
        
        Args:
            input_ids: (batch, seq_len) tokenized prompts
            attention_mask: (batch, seq_len) attention mask
            max_length: Maximum output length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            List of zip file bytes for each item in batch
        """
        self.eval()
        
        # Encode prompt
        encoder_output, encoder_pooled = self.text_encoder(input_ids, attention_mask)
        
        # Generate content bytes
        generated = self.content_decoder.generate(
            encoder_output,
            encoder_pooled,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Convert to bytes
        zip_bytes_list = []
        for seq in generated:
            # Remove BOS/EOS tokens and convert to bytes
            seq = seq[1:]  # Remove BOS
            eos_idx = (seq == 1).nonzero(as_tuple=True)[0]
            if len(eos_idx) > 0:
                seq = seq[:eos_idx[0]]
            
            # Convert token indices to bytes
            byte_values = seq.clamp(0, 255).cpu().numpy().astype('uint8')
            zip_bytes_list.append(bytes(byte_values))
        
        return zip_bytes_list
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ZipPredictorModelLarge(ZipPredictorModel):
    """Large variant of the ZipPredictor model."""
    
    def __init__(self, **kwargs):
        defaults = {
            'd_model': 1024,
            'encoder_layers': 12,
            'decoder_layers': 12,
            'nhead': 16,
            'dim_feedforward': 4096,
            'max_output_len': 131072
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class ZipPredictorModelXL(ZipPredictorModel):
    """Extra-large variant of the ZipPredictor model."""
    
    def __init__(self, **kwargs):
        defaults = {
            'd_model': 2048,
            'encoder_layers': 24,
            'decoder_layers': 24,
            'nhead': 32,
            'dim_feedforward': 8192,
            'max_output_len': 262144
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


# Model registry
MODEL_REGISTRY = {
    'base': ZipPredictorModel,
    'large': ZipPredictorModelLarge,
    'xl': ZipPredictorModelXL
}


def create_model(model_size: str = 'base', **kwargs) -> ZipPredictorModel:
    """
    Factory function to create models.
    
    Args:
        model_size: One of 'base', 'large', 'xl'
        **kwargs: Override default configuration
        
    Returns:
        Instantiated model
    """
    if model_size not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_size](**kwargs)
