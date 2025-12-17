"""
Training Pipeline for Zip Predictor Model

This module provides the training infrastructure to distill knowledge
from a teacher model M (loaded from .safetensors) into our student model K.

Key Components:
- TeacherModelWrapper: Loads and runs the teacher model
- DatasetGenerator: Creates training pairs from teacher outputs
- DistillationTrainer: Handles the training loop with distillation losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import safetensors.torch as st
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
import io
import zipfile
import json
import hashlib
from dataclasses import dataclass, field
from tqdm import tqdm
import logging
import wandb
from datetime import datetime

from .architecture import ZipPredictorModel, create_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model
    model_size: str = 'base'
    
    # Data
    batch_size: int = 8
    num_workers: int = 4
    max_seq_len: int = 512
    max_output_len: int = 65536
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Distillation
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5  # Weight for distillation loss
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = './checkpoints'
    
    # Logging
    log_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = 'zip-predictor'
    
    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class TeacherModelWrapper:
    """
    Wrapper for loading and running the teacher model from safetensors.
    
    This class handles loading arbitrary model architectures from .safetensors
    files and provides a unified interface for generating outputs.
    """
    
    def __init__(
        self,
        safetensors_path: Union[str, Path],
        model_config: Optional[Dict[str, Any]] = None,
        model_class: Optional[type] = None,
        device: str = 'cuda'
    ):
        """
        Initialize teacher model from safetensors file.
        
        Args:
            safetensors_path: Path to .safetensors file
            model_config: Configuration dict for model architecture
            model_class: Optional custom model class to instantiate
            device: Device to load model on
        """
        self.device = device
        self.safetensors_path = Path(safetensors_path)
        
        # Load state dict from safetensors
        logger.info(f"Loading teacher model from {safetensors_path}")
        self.state_dict = st.load_file(str(self.safetensors_path))
        
        # Infer or use provided config
        self.config = model_config or self._infer_config_from_state_dict()
        
        # Build model
        if model_class is not None:
            self.model = model_class(**self.config)
        else:
            self.model = self._build_model_from_state_dict()
        
        # Load weights
        self.model.load_state_dict(self.state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Teacher model loaded with {self._count_params()} parameters")
    
    def _infer_config_from_state_dict(self) -> Dict[str, Any]:
        """Attempt to infer model configuration from state dict shapes."""
        config = {}
        
        for key, tensor in self.state_dict.items():
            # Infer embedding dimensions
            if 'embed' in key.lower() and len(tensor.shape) == 2:
                if 'token' in key.lower() or 'word' in key.lower():
                    config['vocab_size'] = tensor.shape[0]
                    config['d_model'] = tensor.shape[1]
            
            # Infer number of layers
            if 'layer' in key.lower():
                import re
                match = re.search(r'layer[._]?(\d+)', key.lower())
                if match:
                    layer_num = int(match.group(1))
                    config['num_layers'] = max(config.get('num_layers', 0), layer_num + 1)
            
            # Infer attention heads from query projection
            if 'query' in key.lower() or 'q_proj' in key.lower():
                if len(tensor.shape) == 2:
                    config['nhead'] = tensor.shape[0] // (config.get('d_model', tensor.shape[1]) // 8)
        
        return config
    
    def _build_model_from_state_dict(self) -> nn.Module:
        """Build a generic model container from state dict."""
        # This is a fallback - ideally the model class should be provided
        logger.warning("Building generic model container from state dict")
        
        class GenericModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                # Register all parameters
                for key, value in state_dict.items():
                    # Convert dots to module hierarchy
                    parts = key.split('.')
                    self._register_nested_param(parts, value)
            
            def _register_nested_param(self, parts: List[str], value: torch.Tensor):
                current = self
                for i, part in enumerate(parts[:-1]):
                    if not hasattr(current, part):
                        setattr(current, part, nn.Module())
                    current = getattr(current, part)
                setattr(current, parts[-1], nn.Parameter(value))
            
            def forward(self, x):
                raise NotImplementedError("Generic model requires custom forward pass")
        
        return GenericModel(self.state_dict)
    
    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer: Any,
        generate_fn: Optional[Callable] = None,
        **kwargs
    ) -> bytes:
        """
        Generate output from the teacher model.
        
        Args:
            prompt: Input text prompt
            tokenizer: Tokenizer for the model
            generate_fn: Custom generation function
            **kwargs: Additional generation parameters
            
        Returns:
            Output bytes (to be zipped)
        """
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        if generate_fn is not None:
            output = generate_fn(self.model, inputs, **kwargs)
        else:
            # Default generation (assumes model has generate method)
            if hasattr(self.model, 'generate'):
                output = self.model.generate(**inputs, **kwargs)
            else:
                output = self.model(**inputs)
        
        return output


class ZipDataset(Dataset):
    """
    Dataset for training the zip predictor model.
    
    Can be created from:
    1. Pre-generated (prompt, zip_bytes) pairs
    2. On-the-fly generation using teacher model
    """
    
    def __init__(
        self,
        prompts: List[str],
        zip_bytes: Optional[List[bytes]] = None,
        teacher_model: Optional[TeacherModelWrapper] = None,
        tokenizer: Any = None,
        max_seq_len: int = 512,
        max_output_len: int = 65536,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize dataset.
        
        Args:
            prompts: List of input prompts
            zip_bytes: Pre-generated zip bytes (optional)
            teacher_model: Teacher model for on-the-fly generation
            tokenizer: Tokenizer for encoding prompts
            max_seq_len: Maximum input sequence length
            max_output_len: Maximum output length
            cache_dir: Directory to cache generated outputs
        """
        self.prompts = prompts
        self.zip_bytes = zip_bytes
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_output_len = max_output_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate
        if zip_bytes is None and teacher_model is None:
            raise ValueError("Either zip_bytes or teacher_model must be provided")
        
        if zip_bytes is not None and len(zip_bytes) != len(prompts):
            raise ValueError("Number of prompts and zip_bytes must match")
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for an index."""
        prompt_hash = hashlib.md5(self.prompts[idx].encode()).hexdigest()[:16]
        return self.cache_dir / f"{idx}_{prompt_hash}.zip"
    
    def _generate_and_cache(self, idx: int) -> bytes:
        """Generate output for a prompt and cache it."""
        cache_path = self._get_cache_path(idx)
        
        if cache_path.exists():
            return cache_path.read_bytes()
        
        # Generate from teacher
        output = self.teacher_model.generate(
            self.prompts[idx],
            self.tokenizer
        )
        
        # Create zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('output.bin', output if isinstance(output, bytes) else str(output).encode())
        
        zip_bytes = zip_buffer.getvalue()
        
        # Cache
        cache_path.write_bytes(zip_bytes)
        
        return zip_bytes
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        
        # Get zip bytes
        if self.zip_bytes is not None:
            zip_data = self.zip_bytes[idx]
        else:
            zip_data = self._generate_and_cache(idx)
        
        # Tokenize prompt
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                prompt,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            # Fallback: character-level encoding
            input_ids = torch.tensor([ord(c) for c in prompt[:self.max_seq_len]], dtype=torch.long)
            input_ids = F.pad(input_ids, (0, self.max_seq_len - len(input_ids)))
            attention_mask = (input_ids != 0).long()
        
        # Encode zip bytes as target
        target_bytes = torch.tensor(list(zip_data[:self.max_output_len]), dtype=torch.long)
        target_bytes = F.pad(target_bytes, (0, self.max_output_len - len(target_bytes)))
        target_mask = (target_bytes != 0).long()
        
        # Parse zip structure for file metadata
        file_metadata = self._parse_zip_structure(zip_data)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_bytes': target_bytes,
            'target_mask': target_mask,
            'file_count': torch.tensor(file_metadata['file_count'], dtype=torch.long),
            'original_length': torch.tensor(len(zip_data), dtype=torch.long)
        }
    
    def _parse_zip_structure(self, zip_data: bytes) -> Dict[str, Any]:
        """Parse zip file to extract structure metadata."""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
                file_list = zf.namelist()
                return {
                    'file_count': len(file_list),
                    'filenames': file_list,
                    'filesizes': [zf.getinfo(f).file_size for f in file_list]
                }
        except:
            return {'file_count': 1, 'filenames': [], 'filesizes': []}


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    Components:
    1. Hard label loss: Cross-entropy with ground truth
    2. Soft label loss: KL divergence with teacher outputs
    3. File structure loss: Match predicted file structure
    4. Length loss: Predict correct output length
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        length_weight: float = 0.1,
        structure_weight: float = 0.1
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.length_weight = length_weight
        self.structure_weight = structure_weight
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        target_bytes: torch.Tensor,
        target_mask: torch.Tensor,
        predicted_length: torch.Tensor,
        actual_length: torch.Tensor,
        file_count_logits: torch.Tensor,
        actual_file_count: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Returns:
            Dictionary with total loss and component losses
        """
        batch_size = student_logits.size(0)
        
        # Hard label loss
        hard_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            target_bytes.view(-1)
        )
        
        # Soft label loss (if teacher provided)
        if teacher_logits is not None:
            soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            soft_loss = torch.tensor(0.0, device=student_logits.device)
            total_loss = hard_loss
        
        # Length prediction loss
        length_loss = self.mse_loss(
            predicted_length,
            actual_length.float()
        )
        total_loss = total_loss + self.length_weight * length_loss
        
        # File structure loss
        structure_loss = F.cross_entropy(
            file_count_logits,
            actual_file_count.long()
        )
        total_loss = total_loss + self.structure_weight * structure_loss
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
            'length_loss': length_loss,
            'structure_loss': structure_loss
        }


class DistillationTrainer:
    """
    Trainer for distilling knowledge from teacher model M to student model K.
    """
    
    def __init__(
        self,
        model: ZipPredictorModel,
        config: TrainingConfig,
        train_dataset: ZipDataset,
        eval_dataset: Optional[ZipDataset] = None,
        teacher_model: Optional[TeacherModelWrapper] = None
    ):
        self.model = model.to(config.device)
        self.config = config
        self.teacher_model = teacher_model
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.eval_loader = None
        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2
        )
        
        # Loss
        self.criterion = DistillationLoss(
            temperature=config.distillation_temperature,
            alpha=config.distillation_alpha
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None
        
        # Tracking
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.to_dict()
            )
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total steps: {self.config.max_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        
        self.model.train()
        accumulated_loss = 0.0
        
        pbar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                loss_dict = self._training_step(batch)
                accumulated_loss += loss_dict['total_loss'].item()
                
                # Gradient accumulation
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                pbar.update(1)
                
                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_loss = accumulated_loss / self.config.log_steps
                    self._log_metrics({
                        'train/loss': avg_loss,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        **{f'train/{k}': v.item() for k, v in loss_dict.items()}
                    })
                    accumulated_loss = 0.0
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.eval_loader:
                    eval_metrics = self._evaluate()
                    self._log_metrics(eval_metrics)
                    
                    if eval_metrics['eval/loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['eval/loss']
                        self._save_checkpoint('best')
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f'step_{self.global_step}')
        
        pbar.close()
        self._save_checkpoint('final')
        logger.info("Training complete!")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single training step."""
        # Move to device
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        
        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                target_bytes=batch['target_bytes'],
                target_mask=batch['target_mask']
            )
            
            # Compute loss
            loss_dict = self.criterion(
                student_logits=outputs['logits'],
                target_bytes=batch['target_bytes'],
                target_mask=batch['target_mask'],
                predicted_length=outputs['predicted_length'],
                actual_length=batch['original_length'],
                file_count_logits=outputs['file_count_logits'],
                actual_file_count=batch['file_count']
            )
        
        # Backward pass
        loss = loss_dict['total_loss'] / self.config.gradient_accumulation_steps
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss_dict
    
    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_loader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                target_bytes=batch['target_bytes'],
                target_mask=batch['target_mask']
            )
            
            loss_dict = self.criterion(
                student_logits=outputs['logits'],
                target_bytes=batch['target_bytes'],
                target_mask=batch['target_mask'],
                predicted_length=outputs['predicted_length'],
                actual_length=batch['original_length'],
                file_count_logits=outputs['file_count_logits'],
                actual_file_count=batch['file_count']
            )
            
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
        
        self.model.train()
        
        return {'eval/loss': total_loss / num_batches}
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and wandb."""
        logger.info(f"Step {self.global_step}: {metrics}")
        
        if self.config.use_wandb:
            wandb.log(metrics, step=self.global_step)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_{name}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model weights
        torch.save(
            self.model.state_dict(),
            checkpoint_path / "model.pt"
        )
        
        # Save as safetensors
        st.save_file(
            self.model.state_dict(),
            str(checkpoint_path / "model.safetensors")
        )
        
        # Save config
        with open(checkpoint_path / "config.json", 'w') as f:
            json.dump(self.model.config, f, indent=2)
        
        # Save training state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }, checkpoint_path / "training_state.pt")
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model weights
        if (checkpoint_path / "model.safetensors").exists():
            state_dict = st.load_file(str(checkpoint_path / "model.safetensors"))
        else:
            state_dict = torch.load(checkpoint_path / "model.pt")
        
        self.model.load_state_dict(state_dict)
        
        # Load training state
        training_state = torch.load(checkpoint_path / "training_state.pt")
        self.optimizer.load_state_dict(training_state['optimizer'])
        self.scheduler.load_state_dict(training_state['scheduler'])
        self.global_step = training_state['global_step']
        self.best_eval_loss = training_state['best_eval_loss']
        
        if self.scaler and training_state.get('scaler'):
            self.scaler.load_state_dict(training_state['scaler'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def train_from_safetensors(
    safetensors_path: str,
    prompts: List[str],
    model_size: str = 'base',
    output_dir: str = './checkpoints',
    **kwargs
) -> ZipPredictorModel:
    """
    Convenience function to train model K from a safetensors file.
    
    Args:
        safetensors_path: Path to teacher model's .safetensors file
        prompts: List of training prompts
        model_size: Size of student model ('base', 'large', 'xl')
        output_dir: Directory for checkpoints
        **kwargs: Additional training config overrides
        
    Returns:
        Trained ZipPredictorModel
    """
    # Load teacher
    teacher = TeacherModelWrapper(safetensors_path)
    
    # Create student model
    model = create_model(model_size)
    
    # Create config
    config = TrainingConfig(
        model_size=model_size,
        output_dir=output_dir,
        **kwargs
    )
    
    # Create dataset
    dataset = ZipDataset(
        prompts=prompts,
        teacher_model=teacher,
        cache_dir=Path(output_dir) / 'cache'
    )
    
    # Train
    trainer = DistillationTrainer(
        model=model,
        config=config,
        train_dataset=dataset,
        teacher_model=teacher
    )
    
    trainer.train()
    
    return model
