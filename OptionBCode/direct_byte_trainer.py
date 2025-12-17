"""
Trainer for Direct Byte Generation Model (Option B)

This trainer is optimized for the unique challenges of direct byte generation:
1. Very long sequences (up to 512KB)
2. Learning implicit zip format structure
3. Handling variable-length outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, List, Dict, Any, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import time
from tqdm import tqdm
import io
import zipfile

from .direct_byte_architecture import (
    DirectByteGenerator,
    DirectByteModelConfig,
    DirectByteLoss,
    SpecialTokens,
    create_direct_byte_model,
    bytes_to_tokens,
    validate_zip
)

logger = logging.getLogger(__name__)


@dataclass
class DirectByteTrainingConfig:
    """Training configuration for direct byte generation."""
    # Model
    model_size: str = 'base'
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Loss
    label_smoothing: float = 0.1
    length_loss_weight: float = 0.1
    
    # Data
    max_prompt_len: int = 512
    max_output_len: int = 65536
    
    # Optimization
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    
    # Logging & checkpointing
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    output_dir: str = './checkpoints_direct'
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_start_len: int = 1024
    curriculum_end_len: int = 65536
    curriculum_epochs: int = 10


class ByteSequenceDataset(Dataset):
    """
    Dataset for (prompt, zip_bytes) pairs.
    
    Handles:
    - Tokenization of prompts
    - Byte encoding of zip files
    - Padding and masking
    """
    
    def __init__(
        self,
        prompts: List[str],
        zip_bytes_list: List[bytes],
        tokenizer: Any,
        max_prompt_len: int = 512,
        max_output_len: int = 65536
    ):
        assert len(prompts) == len(zip_bytes_list)
        self.prompts = prompts
        self.zip_bytes_list = zip_bytes_list
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_output_len = max_output_len
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        zip_bytes = self.zip_bytes_list[idx]
        
        # Tokenize prompt
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_prompt_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Convert zip bytes to tokens
        byte_tokens = bytes_to_tokens(zip_bytes)
        target_length = len(byte_tokens)
        
        # Truncate if necessary
        if len(byte_tokens) > self.max_output_len:
            byte_tokens = byte_tokens[:self.max_output_len]
            byte_tokens[-1] = SpecialTokens.EOS
        
        # Pad to max length
        padded_bytes = torch.full(
            (self.max_output_len,),
            SpecialTokens.PAD,
            dtype=torch.long
        )
        padded_bytes[:len(byte_tokens)] = byte_tokens
        
        # Create mask
        byte_mask = torch.zeros(self.max_output_len, dtype=torch.float)
        byte_mask[:len(byte_tokens)] = 1.0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_bytes': padded_bytes,
            'target_mask': byte_mask,
            'target_length': torch.tensor(target_length, dtype=torch.float)
        }


class StreamingTeacherDataset(Dataset):
    """
    Dataset that generates training data on-the-fly from teacher model.
    
    This avoids storing all zip outputs in memory.
    """
    
    def __init__(
        self,
        prompts: List[str],
        teacher_generate_fn: Callable[[str], bytes],
        tokenizer: Any,
        max_prompt_len: int = 512,
        max_output_len: int = 65536,
        cache_dir: Optional[str] = None
    ):
        self.prompts = prompts
        self.teacher_generate_fn = teacher_generate_fn
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_output_len = max_output_len
        
        # Optional caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def _get_cached_or_generate(self, idx: int) -> bytes:
        """Get from cache or generate using teacher."""
        if self.cache_dir:
            cache_file = self.cache_dir / f"sample_{idx}.zip"
            if cache_file.exists():
                return cache_file.read_bytes()
        
        # Generate using teacher
        zip_bytes = self.teacher_generate_fn(self.prompts[idx])
        
        # Cache if enabled
        if self.cache_dir:
            cache_file.write_bytes(zip_bytes)
        
        return zip_bytes
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        zip_bytes = self._get_cached_or_generate(idx)
        
        # Same processing as ByteSequenceDataset
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_prompt_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        byte_tokens = bytes_to_tokens(zip_bytes)
        target_length = len(byte_tokens)
        
        if len(byte_tokens) > self.max_output_len:
            byte_tokens = byte_tokens[:self.max_output_len]
            byte_tokens[-1] = SpecialTokens.EOS
        
        padded_bytes = torch.full(
            (self.max_output_len,),
            SpecialTokens.PAD,
            dtype=torch.long
        )
        padded_bytes[:len(byte_tokens)] = byte_tokens
        
        byte_mask = torch.zeros(self.max_output_len, dtype=torch.float)
        byte_mask[:len(byte_tokens)] = 1.0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_bytes': padded_bytes,
            'target_mask': byte_mask,
            'target_length': torch.tensor(target_length, dtype=torch.float)
        }


class CurriculumSampler:
    """
    Implements curriculum learning by gradually increasing sequence length.
    
    Starts with shorter sequences and progressively includes longer ones.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        lengths: List[int],
        start_len: int = 1024,
        end_len: int = 65536,
        epochs_to_full: int = 10
    ):
        self.dataset = dataset
        self.lengths = lengths
        self.start_len = start_len
        self.end_len = end_len
        self.epochs_to_full = epochs_to_full
        
        # Sort indices by length
        self.sorted_indices = sorted(
            range(len(lengths)),
            key=lambda i: lengths[i]
        )
    
    def get_indices_for_epoch(self, epoch: int) -> List[int]:
        """Get sample indices for given epoch based on curriculum."""
        # Calculate current max length
        progress = min(epoch / self.epochs_to_full, 1.0)
        current_max_len = int(
            self.start_len + progress * (self.end_len - self.start_len)
        )
        
        # Filter indices
        valid_indices = [
            idx for idx in self.sorted_indices
            if self.lengths[idx] <= current_max_len
        ]
        
        return valid_indices


class DirectByteTrainer:
    """
    Trainer for Direct Byte Generation models.
    
    Features:
    - Mixed precision training
    - Gradient accumulation for large effective batch sizes
    - Curriculum learning for handling long sequences
    - Validation with zip validity checking
    """
    
    def __init__(
        self,
        model: DirectByteGenerator,
        config: DirectByteTrainingConfig,
        tokenizer: Any
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Enable gradient checkpointing if configured
        if config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Loss function
        self.loss_fn = DirectByteLoss(
            label_smoothing=config.label_smoothing,
            length_loss_weight=config.length_loss_weight
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Gradient scaler for AMP
        self.scaler = GradScaler() if config.use_amp else None
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        for layer in self.model.encoder.layers:
            layer.forward = torch.utils.checkpoint.checkpoint(
                layer.forward,
                use_reentrant=False
            )
        for layer in self.model.decoder.layers:
            layer.forward = torch.utils.checkpoint.checkpoint(
                layer.forward,
                use_reentrant=False
            )
    
    def _get_lr_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, num_training_steps - self.config.warmup_steps
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with optional AMP
        with autocast(enabled=self.config.use_amp):
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                target_bytes=batch['target_bytes']
            )
            
            losses = self.loss_fn(
                outputs['logits'],
                batch['target_bytes'],
                outputs['predicted_length'],
                batch['target_length']
            )
            
            loss = losses['total_loss'] / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {
            'loss': losses['total_loss'].item(),
            'ce_loss': losses['ce_loss'].item(),
            'length_loss': losses['length_loss'].item()
        }
    
    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.global_step += 1
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_dataloader: DataLoader,
        num_samples: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Metrics:
        - Average loss
        - Zip validity rate
        - Length prediction accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_length_loss = 0.0
        num_batches = 0
        
        valid_zips = 0
        total_generated = 0
        length_errors = []
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Compute loss
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                target_bytes=batch['target_bytes']
            )
            
            losses = self.loss_fn(
                outputs['logits'],
                batch['target_bytes'],
                outputs['predicted_length'],
                batch['target_length']
            )
            
            total_loss += losses['total_loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            total_length_loss += losses['length_loss'].item()
            num_batches += 1
            
            # Generate samples and check validity
            if total_generated < num_samples:
                generated = self.model.generate(
                    input_ids=batch['input_ids'][:1],
                    attention_mask=batch['attention_mask'][:1],
                    max_length=10000  # Limit for eval speed
                )
                
                for zip_bytes in generated:
                    if validate_zip(zip_bytes):
                        valid_zips += 1
                    total_generated += 1
                
                # Track length prediction accuracy
                length_errors.append(
                    abs(outputs['predicted_length'][0].item() - batch['target_length'][0].item())
                )
        
        self.model.train()
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_ce_loss': total_ce_loss / num_batches,
            'eval_length_loss': total_length_loss / num_batches,
            'zip_validity_rate': valid_zips / max(1, total_generated),
            'avg_length_error': sum(length_errors) / max(1, len(length_errors))
        }
    
    def save_checkpoint(self, name: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'metrics': metrics
        }
        
        path = self.output_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        # Also save as safetensors
        try:
            from safetensors.torch import save_file
            safetensors_path = self.output_dir / f"{name}.safetensors"
            save_file(self.model.state_dict(), safetensors_path)
        except ImportError:
            pass
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        logger.info(f"Loaded checkpoint from {path}")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ):
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional validation dataset
        """
        import math
        
        # Create data loader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Calculate training steps
        steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.max_epochs
        
        # Learning rate scheduler
        scheduler = self._get_lr_scheduler(total_steps)
        
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Model parameters: {self.model.get_num_params():,}")
        
        # Training loop
        self.model.train()
        accumulation_loss = 0.0
        
        for epoch in range(self.config.max_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.max_epochs}"
            )
            
            for step, batch in enumerate(progress_bar):
                # Training step
                step_losses = self.train_step(batch)
                accumulation_loss += step_losses['loss']
                
                # Optimizer step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    scheduler.step()
                    
                    epoch_loss += accumulation_loss
                    epoch_steps += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{accumulation_loss:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    accumulation_loss = 0.0
                    
                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        logger.info(
                            f"Step {self.global_step}: "
                            f"loss={step_losses['loss']:.4f}, "
                            f"ce_loss={step_losses['ce_loss']:.4f}, "
                            f"len_loss={step_losses['length_loss']:.4f}"
                        )
                    
                    # Evaluation
                    if eval_dataloader and self.global_step % self.config.eval_every == 0:
                        eval_metrics = self.evaluate(eval_dataloader)
                        logger.info(f"Eval metrics: {eval_metrics}")
                        
                        # Save best model
                        if eval_metrics['eval_loss'] < self.best_loss:
                            self.best_loss = eval_metrics['eval_loss']
                            self.save_checkpoint('best', eval_metrics)
                    
                    # Periodic checkpointing
                    if self.global_step % self.config.save_every == 0:
                        self.save_checkpoint(f'step_{self.global_step}')
            
            # End of epoch
            avg_epoch_loss = epoch_loss / max(1, epoch_steps)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}')
        
        # Final save
        self.save_checkpoint('final')
        logger.info("Training completed!")


def train_direct_byte_model(
    teacher_path: str,
    prompts: List[str],
    teacher_generate_fn: Callable[[str], bytes],
    tokenizer: Any,
    config: Optional[DirectByteTrainingConfig] = None,
    eval_prompts: Optional[List[str]] = None
) -> DirectByteGenerator:
    """
    High-level function to train a Direct Byte Generator.
    
    Args:
        teacher_path: Path to teacher model (for logging)
        prompts: Training prompts
        teacher_generate_fn: Function to generate zip from prompt using teacher
        tokenizer: Text tokenizer
        config: Training configuration
        eval_prompts: Optional validation prompts
        
    Returns:
        Trained model
    """
    config = config or DirectByteTrainingConfig()
    
    # Create model
    model = create_direct_byte_model(config.model_size)
    logger.info(f"Created {config.model_size} model with {model.get_num_params():,} parameters")
    
    # Create datasets
    train_dataset = StreamingTeacherDataset(
        prompts=prompts,
        teacher_generate_fn=teacher_generate_fn,
        tokenizer=tokenizer,
        max_prompt_len=config.max_prompt_len,
        max_output_len=config.max_output_len,
        cache_dir=str(Path(config.output_dir) / 'cache')
    )
    
    eval_dataset = None
    if eval_prompts:
        eval_dataset = StreamingTeacherDataset(
            prompts=eval_prompts,
            teacher_generate_fn=teacher_generate_fn,
            tokenizer=tokenizer,
            max_prompt_len=config.max_prompt_len,
            max_output_len=config.max_output_len
        )
    
    # Create trainer
    trainer = DirectByteTrainer(model, config, tokenizer)
    
    # Train
    trainer.train(train_dataset, eval_dataset)
    
    return model


# Example usage
if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser(description='Train Direct Byte Generator')
    parser.add_argument('--prompts', type=str, required=True, help='Path to prompts file')
    parser.add_argument('--output-dir', type=str, default='./checkpoints_direct')
    parser.add_argument('--model-size', type=str, default='base', choices=['small', 'base', 'large', 'xl'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    
    # Load prompts
    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create config
    config = DirectByteTrainingConfig(
        model_size=args.model_size,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # Note: You would need to implement teacher_generate_fn
    # This is a placeholder that creates empty zips
    def dummy_teacher(prompt: str) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            zf.writestr('output.txt', f'Response to: {prompt}')
        return buffer.getvalue()
    
    # Train
    model = train_direct_byte_model(
        teacher_path='dummy',
        prompts=prompts,
        teacher_generate_fn=dummy_teacher,
        tokenizer=tokenizer,
        config=config
    )
