"""
Option B: Direct Byte Generation Architecture

This module implements a simpler but more challenging approach where
the model directly generates raw zip file bytes autoregressively,
without intermediate structure prediction.

Architecture:
    Prompt → [Encoder] → [Autoregressive Byte Decoder] → Raw Bytes → Zip File

Pros:
- Simpler architecture, fewer components
- End-to-end differentiable
- Can learn arbitrary binary patterns

Cons:
- Must learn zip format structure implicitly
- Harder to train (longer sequences, subtle patterns)
- More prone to producing invalid zip files
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import math
from dataclasses import dataclass
from enum import IntEnum


class SpecialTokens(IntEnum):
    """Special tokens for byte-level generation."""
    PAD = 256
    BOS = 257  # Beginning of sequence
    EOS = 258  # End of sequence
    UNK = 259
    
    @classmethod
    def vocab_size(cls) -> int:
        return 260  # 256 bytes + 4 special tokens


@dataclass
class DirectByteModelConfig:
    """Configuration for Direct Byte Generation model."""
    # Encoder config
    text_vocab_size: int = 50257  # GPT-2 tokenizer
    d_model: int = 768
    encoder_layers: int = 6
    encoder_heads: int = 12
    encoder_ff_dim: int = 3072
    max_prompt_len: int = 512
    
    # Decoder config
    byte_vocab_size: int = 260  # 256 bytes + special tokens
    decoder_layers: int = 12
    decoder_heads: int = 12
    decoder_ff_dim: int = 3072
    max_output_len: int = 131072  # 128KB max zip size
    
    # Training config
    dropout: float = 0.1
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    
    # Generation config
    default_temperature: float = 0.8
    default_top_k: int = 50
    default_top_p: float = 0.95


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better length generalization.
    
    This is particularly important for byte-level generation where
    sequences can be very long.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 131072, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE and causal masking."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        is_causal: bool = False,
        use_rope: bool = True,
        max_seq_len: int = 131072
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal
        self.use_rope = use_rope
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # Project
        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.use_rope:
            cos, sin = self.rope(q, max(tgt_len, src_len))
            q, k = apply_rotary_pos_emb(q, k, cos[:, :, :tgt_len], cos[:, :, :src_len])
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim)
        self.w2 = nn.Linear(ff_dim, d_model)
        self.w3 = nn.Linear(d_model, ff_dim)  # For gating
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(xW1) * (xW3)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, is_causal=False, use_rope=False)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(x)
        
        # Feed forward
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer with causal self-attention and cross-attention."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 131072
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout, 
            is_causal=True, use_rope=True, max_seq_len=max_seq_len
        )
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, dropout,
            is_causal=False, use_rope=False
        )
        self.ff = FeedForward(d_model, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Causal self attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x)
        x = residual + self.dropout(x)
        
        # Cross attention to encoder
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, encoder_output, encoder_output, key_padding_mask=encoder_padding_mask)
        x = residual + self.dropout(x)
        
        # Feed forward
        residual = x
        x = self.norm3(x)
        x = residual + self.ff(x)
        
        return x


class PromptEncoder(nn.Module):
    """
    Encodes text prompts into contextual representations.
    
    Uses a standard transformer encoder with learned positional embeddings.
    """
    
    def __init__(self, config: DirectByteModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.text_vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_prompt_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(
                config.d_model,
                config.encoder_heads,
                config.encoder_ff_dim,
                config.dropout
            )
            for _ in range(config.encoder_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding
            
        Returns:
            encoder_output: (batch, seq_len, d_model)
            pooled_output: (batch, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Create padding mask for attention
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        # Encoder layers
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        
        x = self.final_norm(x)
        
        # Pooled output (mean of non-padding tokens)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        
        return x, pooled


class ByteDecoder(nn.Module):
    """
    Autoregressively generates bytes conditioned on encoder output.
    
    Uses causal self-attention and cross-attention to the prompt encoding.
    Outputs logits over 260 classes (256 bytes + 4 special tokens).
    """
    
    def __init__(self, config: DirectByteModelConfig):
        super().__init__()
        self.config = config
        
        self.byte_embedding = nn.Embedding(config.byte_vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(
                config.d_model,
                config.decoder_heads,
                config.decoder_ff_dim,
                config.dropout,
                config.max_output_len
            )
            for _ in range(config.decoder_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.byte_vocab_size)
        
        # Length predictor (auxiliary task)
        self.length_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Softplus()
        )
    
    def forward(
        self,
        target_bytes: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_pooled: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with teacher forcing.
        
        Args:
            target_bytes: (batch, tgt_len) byte indices (including BOS)
            encoder_output: (batch, src_len, d_model)
            encoder_pooled: (batch, d_model)
            encoder_padding_mask: (batch, src_len)
            
        Returns:
            Dict with logits and predicted_length
        """
        # Embed target bytes
        x = self.byte_embedding(target_bytes)
        x = self.dropout(x)
        
        # Convert attention mask to key padding mask
        if encoder_padding_mask is not None:
            key_padding_mask = ~encoder_padding_mask.bool()
        else:
            key_padding_mask = None
        
        # Decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, key_padding_mask)
        
        x = self.final_norm(x)
        
        # Output logits
        logits = self.output_projection(x)
        
        # Predict length from encoder
        predicted_length = self.length_predictor(encoder_pooled).squeeze(-1)
        
        return {
            'logits': logits,
            'predicted_length': predicted_length
        }
    
    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_pooled: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressively generate bytes.
        
        Args:
            encoder_output: (batch, src_len, d_model)
            encoder_pooled: (batch, d_model)
            encoder_padding_mask: (batch, src_len)
            max_length: Maximum bytes to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated bytes
            
        Returns:
            generated: (batch, gen_len) generated byte indices
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Predict length if not specified
        if max_length is None:
            predicted_len = self.length_predictor(encoder_pooled).squeeze(-1)
            max_length = min(int(predicted_len.max().item() * 1.5), self.config.max_output_len)
        
        # Start with BOS token
        generated = torch.full(
            (batch_size, 1),
            SpecialTokens.BOS,
            dtype=torch.long,
            device=device
        )
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Convert padding mask
        if encoder_padding_mask is not None:
            key_padding_mask = ~encoder_padding_mask.bool()
        else:
            key_padding_mask = None
        
        # KV cache for efficient generation
        past_key_values = None
        
        for step in range(max_length):
            # Get embeddings for current tokens
            if past_key_values is None:
                x = self.byte_embedding(generated)
            else:
                x = self.byte_embedding(generated[:, -1:])
            
            x = self.dropout(x)
            
            # Run through decoder (simplified - full implementation would use KV cache)
            for layer in self.layers:
                x = layer(x if past_key_values is None else x, encoder_output, key_padding_mask)
            
            x = self.final_norm(x)
            
            # Get logits for last position
            logits = self.output_projection(x[:, -1, :])
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for prev_token in generated[i].unique():
                        logits[i, prev_token] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply nucleus (top-p) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == SpecialTokens.EOS)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences finished
            if finished.all():
                break
        
        return generated


class DirectByteGenerator(nn.Module):
    """
    Complete Direct Byte Generation model.
    
    Architecture:
        Prompt → [PromptEncoder] → [ByteDecoder] → Raw Zip Bytes
    
    This model directly generates the raw bytes of a zip file
    autoregressively, learning the zip format structure implicitly.
    """
    
    def __init__(self, config: Optional[DirectByteModelConfig] = None):
        super().__init__()
        self.config = config or DirectByteModelConfig()
        
        self.encoder = PromptEncoder(self.config)
        self.decoder = ByteDecoder(self.config)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using scaled initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_bytes: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_ids: (batch, prompt_len) tokenized prompts
            attention_mask: (batch, prompt_len) attention mask
            target_bytes: (batch, output_len) target byte sequence
            target_mask: (batch, output_len) mask for target padding
            
        Returns:
            Dict containing logits, predicted_length, and encoder outputs
        """
        # Encode prompt
        encoder_output, encoder_pooled = self.encoder(input_ids, attention_mask)
        
        if target_bytes is not None:
            # Training mode
            decoder_output = self.decoder(
                target_bytes,
                encoder_output,
                encoder_pooled,
                attention_mask
            )
            
            return {
                'logits': decoder_output['logits'],
                'predicted_length': decoder_output['predicted_length'],
                'encoder_output': encoder_output,
                'encoder_pooled': encoder_pooled
            }
        else:
            # Just return encoder outputs for generation
            return {
                'encoder_output': encoder_output,
                'encoder_pooled': encoder_pooled
            }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0
    ) -> List[bytes]:
        """
        Generate zip file bytes from prompts.
        
        Args:
            input_ids: (batch, prompt_len) tokenized prompts
            attention_mask: (batch, prompt_len) attention mask
            max_length: Maximum bytes to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            repetition_penalty: Penalty for repeated tokens
            
        Returns:
            List of bytes objects, one per batch item
        """
        self.eval()
        
        # Encode
        encoder_output, encoder_pooled = self.encoder(input_ids, attention_mask)
        
        # Generate
        generated = self.decoder.generate(
            encoder_output,
            encoder_pooled,
            attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        # Convert to bytes
        results = []
        for seq in generated:
            # Remove BOS
            seq = seq[1:]
            
            # Find EOS
            eos_positions = (seq == SpecialTokens.EOS).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                seq = seq[:eos_positions[0]]
            
            # Remove any remaining special tokens and clamp to byte range
            mask = seq < 256
            seq = seq[mask]
            
            # Convert to bytes
            byte_values = seq.cpu().numpy().astype('uint8')
            results.append(bytes(byte_values))
        
        return results
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.token_embedding.weight.numel()
            n_params -= self.encoder.position_embedding.weight.numel()
            n_params -= self.decoder.byte_embedding.weight.numel()
        return n_params


class DirectByteLoss(nn.Module):
    """
    Loss function for direct byte generation.
    
    Combines:
    1. Cross-entropy loss on byte predictions
    2. Length prediction loss
    3. Optional label smoothing
    """
    
    def __init__(
        self,
        label_smoothing: float = 0.1,
        length_loss_weight: float = 0.1,
        ignore_index: int = SpecialTokens.PAD
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        self.length_loss = nn.SmoothL1Loss()
        self.length_loss_weight = length_loss_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        predicted_length: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            logits: (batch, seq_len, vocab_size) predicted logits
            targets: (batch, seq_len) target byte indices
            predicted_length: (batch,) predicted lengths
            target_lengths: (batch,) actual lengths
            
        Returns:
            Dict with total_loss, ce_loss, length_loss
        """
        # Reshape for cross entropy
        batch_size, seq_len, vocab_size = logits.shape
        
        # Shift: predict next byte from current
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        
        ce_loss = self.ce_loss(
            shift_logits.view(-1, vocab_size),
            shift_targets.view(-1)
        )
        
        # Length loss
        length_loss = self.length_loss(
            predicted_length,
            target_lengths.float()
        )
        
        total_loss = ce_loss + self.length_loss_weight * length_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'length_loss': length_loss
        }


# Pre-configured model sizes
class DirectByteGeneratorSmall(DirectByteGenerator):
    """Small model (~50M params) for testing and lightweight use."""
    
    def __init__(self, **kwargs):
        config = DirectByteModelConfig(
            d_model=512,
            encoder_layers=4,
            encoder_heads=8,
            encoder_ff_dim=2048,
            decoder_layers=6,
            decoder_heads=8,
            decoder_ff_dim=2048,
            max_output_len=65536
        )
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        super().__init__(config)


class DirectByteGeneratorBase(DirectByteGenerator):
    """Base model (~125M params)."""
    
    def __init__(self, **kwargs):
        config = DirectByteModelConfig(
            d_model=768,
            encoder_layers=6,
            encoder_heads=12,
            encoder_ff_dim=3072,
            decoder_layers=12,
            decoder_heads=12,
            decoder_ff_dim=3072,
            max_output_len=131072
        )
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        super().__init__(config)


class DirectByteGeneratorLarge(DirectByteGenerator):
    """Large model (~350M params)."""
    
    def __init__(self, **kwargs):
        config = DirectByteModelConfig(
            d_model=1024,
            encoder_layers=12,
            encoder_heads=16,
            encoder_ff_dim=4096,
            decoder_layers=18,
            decoder_heads=16,
            decoder_ff_dim=4096,
            max_output_len=262144
        )
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        super().__init__(config)


class DirectByteGeneratorXL(DirectByteGenerator):
    """Extra-large model (~1B params)."""
    
    def __init__(self, **kwargs):
        config = DirectByteModelConfig(
            d_model=2048,
            encoder_layers=24,
            encoder_heads=32,
            encoder_ff_dim=8192,
            decoder_layers=24,
            decoder_heads=32,
            decoder_ff_dim=8192,
            max_output_len=524288
        )
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        super().__init__(config)


# Model registry
DIRECT_BYTE_MODEL_REGISTRY = {
    'small': DirectByteGeneratorSmall,
    'base': DirectByteGeneratorBase,
    'large': DirectByteGeneratorLarge,
    'xl': DirectByteGeneratorXL
}


def create_direct_byte_model(
    model_size: str = 'base',
    **kwargs
) -> DirectByteGenerator:
    """
    Factory function to create Direct Byte Generator models.
    
    Args:
        model_size: One of 'small', 'base', 'large', 'xl'
        **kwargs: Override default configuration
        
    Returns:
        Instantiated model
    """
    if model_size not in DIRECT_BYTE_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Choose from {list(DIRECT_BYTE_MODEL_REGISTRY.keys())}"
        )
    
    return DIRECT_BYTE_MODEL_REGISTRY[model_size](**kwargs)


# Utility functions for zip handling
def bytes_to_tokens(data: bytes) -> torch.Tensor:
    """Convert bytes to token indices with BOS/EOS."""
    tokens = [SpecialTokens.BOS] + list(data) + [SpecialTokens.EOS]
    return torch.tensor(tokens, dtype=torch.long)


def tokens_to_bytes(tokens: torch.Tensor) -> bytes:
    """Convert token indices back to bytes."""
    # Remove special tokens
    tokens = tokens[(tokens >= 0) & (tokens < 256)]
    return bytes(tokens.cpu().numpy().astype('uint8'))


def validate_zip(data: bytes) -> bool:
    """Check if bytes represent a valid zip file."""
    import zipfile
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
            # Try to read the file list
            zf.namelist()
            return True
    except:
        return False
