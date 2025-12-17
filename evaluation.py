"""
Evaluation Metrics for Zip Predictor Model

Provides comprehensive evaluation of generated zip files including:
- Byte-level accuracy
- Structural similarity
- Content-aware metrics
- Perceptual metrics for specific content types
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import io
import zipfile
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # Byte-level metrics
    byte_accuracy: float = 0.0
    byte_f1: float = 0.0
    edit_distance: float = 0.0
    
    # Structure metrics
    file_count_accuracy: float = 0.0
    filename_similarity: float = 0.0
    structure_f1: float = 0.0
    
    # Content metrics
    content_hash_match: float = 0.0
    content_similarity: float = 0.0
    
    # Validity metrics
    valid_zip_rate: float = 0.0
    decompression_success_rate: float = 0.0
    
    # Per-file metrics
    per_file_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    def summary(self) -> str:
        return f"""
Evaluation Results:
==================
Byte-level Metrics:
  - Accuracy: {self.byte_accuracy:.4f}
  - F1 Score: {self.byte_f1:.4f}
  - Edit Distance: {self.edit_distance:.4f}

Structure Metrics:
  - File Count Accuracy: {self.file_count_accuracy:.4f}
  - Filename Similarity: {self.filename_similarity:.4f}
  - Structure F1: {self.structure_f1:.4f}

Content Metrics:
  - Hash Match Rate: {self.content_hash_match:.4f}
  - Content Similarity: {self.content_similarity:.4f}

Validity Metrics:
  - Valid Zip Rate: {self.valid_zip_rate:.4f}
  - Decompression Success: {self.decompression_success_rate:.4f}
"""


class ByteLevelMetrics:
    """Compute byte-level comparison metrics."""
    
    @staticmethod
    def accuracy(pred: bytes, target: bytes) -> float:
        """Compute byte-level accuracy."""
        min_len = min(len(pred), len(target))
        if min_len == 0:
            return 0.0
        
        matches = sum(p == t for p, t in zip(pred[:min_len], target[:min_len]))
        return matches / max(len(pred), len(target))
    
    @staticmethod
    def edit_distance(pred: bytes, target: bytes) -> int:
        """Compute Levenshtein edit distance."""
        m, n = len(pred), len(target)
        
        # Optimize for large sequences
        if m > 10000 or n > 10000:
            return ByteLevelMetrics._approximate_edit_distance(pred, target)
        
        # Standard DP solution
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def _approximate_edit_distance(pred: bytes, target: bytes) -> int:
        """Approximate edit distance for large sequences using sampling."""
        sample_size = 1000
        
        if len(pred) <= sample_size and len(target) <= sample_size:
            return ByteLevelMetrics.edit_distance(pred, target)
        
        # Sample positions
        pred_samples = np.linspace(0, len(pred)-1, sample_size, dtype=int)
        target_samples = np.linspace(0, len(target)-1, sample_size, dtype=int)
        
        pred_sampled = bytes([pred[i] for i in pred_samples])
        target_sampled = bytes([target[i] for i in target_samples])
        
        sampled_dist = ByteLevelMetrics.edit_distance(pred_sampled, target_sampled)
        
        # Scale to original size
        scale = max(len(pred), len(target)) / sample_size
        return int(sampled_dist * scale)
    
    @staticmethod
    def normalized_edit_distance(pred: bytes, target: bytes) -> float:
        """Compute normalized edit distance (0 = identical, 1 = completely different)."""
        dist = ByteLevelMetrics.edit_distance(pred, target)
        max_len = max(len(pred), len(target))
        return dist / max_len if max_len > 0 else 0.0
    
    @staticmethod
    def byte_f1(pred: bytes, target: bytes) -> float:
        """Compute F1 score treating bytes as a set."""
        pred_set = set(enumerate(pred))
        target_set = set(enumerate(target))
        
        intersection = len(pred_set & target_set)
        
        precision = intersection / len(pred_set) if pred_set else 0.0
        recall = intersection / len(target_set) if target_set else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def compression_ratio_similarity(pred: bytes, target: bytes) -> float:
        """Compare compression characteristics."""
        import zlib
        
        pred_compressed = len(zlib.compress(pred))
        target_compressed = len(zlib.compress(target))
        
        pred_ratio = pred_compressed / len(pred) if pred else 0
        target_ratio = target_compressed / len(target) if target else 0
        
        if pred_ratio + target_ratio == 0:
            return 1.0
        
        return 1 - abs(pred_ratio - target_ratio) / max(pred_ratio, target_ratio)


class StructureMetrics:
    """Compute zip structure comparison metrics."""
    
    @staticmethod
    def parse_structure(zip_data: bytes) -> Dict[str, Any]:
        """Extract structure from zip file."""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
                return {
                    'files': zf.namelist(),
                    'file_info': {
                        name: {
                            'size': zf.getinfo(name).file_size,
                            'compressed_size': zf.getinfo(name).compress_size,
                            'compression': zf.getinfo(name).compress_type
                        }
                        for name in zf.namelist()
                    }
                }
        except:
            return {'files': [], 'file_info': {}}
    
    @staticmethod
    def file_count_accuracy(pred_struct: Dict, target_struct: Dict) -> float:
        """Check if file count matches."""
        pred_count = len(pred_struct.get('files', []))
        target_count = len(target_struct.get('files', []))
        return 1.0 if pred_count == target_count else 0.0
    
    @staticmethod
    def filename_similarity(pred_struct: Dict, target_struct: Dict) -> float:
        """Compute similarity of filenames using Jaccard index."""
        pred_files = set(pred_struct.get('files', []))
        target_files = set(target_struct.get('files', []))
        
        if not pred_files and not target_files:
            return 1.0
        if not pred_files or not target_files:
            return 0.0
        
        intersection = len(pred_files & target_files)
        union = len(pred_files | target_files)
        
        return intersection / union
    
    @staticmethod
    def structure_f1(pred_struct: Dict, target_struct: Dict) -> float:
        """Compute F1 score for file structure."""
        pred_files = set(pred_struct.get('files', []))
        target_files = set(target_struct.get('files', []))
        
        if not pred_files and not target_files:
            return 1.0
        
        true_positives = len(pred_files & target_files)
        
        precision = true_positives / len(pred_files) if pred_files else 0.0
        recall = true_positives / len(target_files) if target_files else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def directory_structure_similarity(pred_struct: Dict, target_struct: Dict) -> float:
        """Compare directory tree structure."""
        def get_directories(files):
            dirs = set()
            for f in files:
                parts = f.split('/')
                for i in range(len(parts) - 1):
                    dirs.add('/'.join(parts[:i+1]))
            return dirs
        
        pred_dirs = get_directories(pred_struct.get('files', []))
        target_dirs = get_directories(target_struct.get('files', []))
        
        if not pred_dirs and not target_dirs:
            return 1.0
        if not pred_dirs or not target_dirs:
            return 0.0
        
        intersection = len(pred_dirs & target_dirs)
        union = len(pred_dirs | target_dirs)
        
        return intersection / union


class ContentMetrics:
    """Compute content-aware comparison metrics."""
    
    @staticmethod
    def content_hash_match(
        pred_data: bytes,
        target_data: bytes
    ) -> Tuple[float, Dict[str, bool]]:
        """Check if file contents match by hash."""
        try:
            pred_zf = zipfile.ZipFile(io.BytesIO(pred_data), 'r')
            target_zf = zipfile.ZipFile(io.BytesIO(target_data), 'r')
            
            matches = {}
            total_matches = 0
            total_files = 0
            
            for name in target_zf.namelist():
                if name in pred_zf.namelist():
                    pred_hash = hashlib.md5(pred_zf.read(name)).hexdigest()
                    target_hash = hashlib.md5(target_zf.read(name)).hexdigest()
                    matches[name] = pred_hash == target_hash
                    total_matches += int(matches[name])
                else:
                    matches[name] = False
                total_files += 1
            
            rate = total_matches / total_files if total_files > 0 else 0.0
            return rate, matches
            
        except:
            return 0.0, {}
    
    @staticmethod
    def content_similarity(
        pred_data: bytes,
        target_data: bytes
    ) -> Tuple[float, Dict[str, float]]:
        """Compute content similarity for each file."""
        try:
            pred_zf = zipfile.ZipFile(io.BytesIO(pred_data), 'r')
            target_zf = zipfile.ZipFile(io.BytesIO(target_data), 'r')
            
            similarities = {}
            total_sim = 0.0
            count = 0
            
            for name in target_zf.namelist():
                if name in pred_zf.namelist():
                    pred_content = pred_zf.read(name)
                    target_content = target_zf.read(name)
                    
                    sim = ByteLevelMetrics.accuracy(pred_content, target_content)
                    similarities[name] = sim
                    total_sim += sim
                    count += 1
                else:
                    similarities[name] = 0.0
            
            avg_sim = total_sim / count if count > 0 else 0.0
            return avg_sim, similarities
            
        except:
            return 0.0, {}
    
    @staticmethod
    def semantic_similarity(
        pred_data: bytes,
        target_data: bytes,
        content_type: str = 'text'
    ) -> float:
        """Compute semantic similarity based on content type."""
        try:
            pred_zf = zipfile.ZipFile(io.BytesIO(pred_data), 'r')
            target_zf = zipfile.ZipFile(io.BytesIO(target_data), 'r')
            
            if content_type == 'text':
                return ContentMetrics._text_semantic_similarity(pred_zf, target_zf)
            elif content_type == 'image':
                return ContentMetrics._image_semantic_similarity(pred_zf, target_zf)
            else:
                return 0.0
        except:
            return 0.0
    
    @staticmethod
    def _text_semantic_similarity(pred_zf: zipfile.ZipFile, target_zf: zipfile.ZipFile) -> float:
        """Compute text semantic similarity using word overlap."""
        def extract_words(zf):
            words = set()
            for name in zf.namelist():
                if name.endswith(('.txt', '.md', '.json', '.xml', '.html')):
                    try:
                        content = zf.read(name).decode('utf-8', errors='ignore')
                        words.update(content.lower().split())
                    except:
                        pass
            return words
        
        pred_words = extract_words(pred_zf)
        target_words = extract_words(target_zf)
        
        if not pred_words and not target_words:
            return 1.0
        if not pred_words or not target_words:
            return 0.0
        
        intersection = len(pred_words & target_words)
        union = len(pred_words | target_words)
        
        return intersection / union
    
    @staticmethod
    def _image_semantic_similarity(pred_zf: zipfile.ZipFile, target_zf: zipfile.ZipFile) -> float:
        """Compute image similarity using structural similarity."""
        try:
            from PIL import Image
            from skimage.metrics import structural_similarity as ssim
            import numpy as np
            
            image_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
            similarities = []
            
            for name in target_zf.namelist():
                if name.lower().endswith(image_exts) and name in pred_zf.namelist():
                    pred_img = Image.open(io.BytesIO(pred_zf.read(name))).convert('RGB')
                    target_img = Image.open(io.BytesIO(target_zf.read(name))).convert('RGB')
                    
                    # Resize to same dimensions
                    target_size = target_img.size
                    pred_img = pred_img.resize(target_size)
                    
                    pred_arr = np.array(pred_img)
                    target_arr = np.array(target_img)
                    
                    sim = ssim(pred_arr, target_arr, channel_axis=2, data_range=255)
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
        except ImportError:
            logger.warning("skimage not available for image similarity")
            return 0.0
        except:
            return 0.0


class ValidityMetrics:
    """Compute zip validity metrics."""
    
    @staticmethod
    def is_valid_zip(data: bytes) -> bool:
        """Check if data is a valid zip file."""
        try:
            with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
                return zf.testzip() is None
        except:
            return False
    
    @staticmethod
    def decompression_test(data: bytes) -> Tuple[bool, Optional[str]]:
        """Test if zip can be fully decompressed."""
        try:
            with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
                for name in zf.namelist():
                    _ = zf.read(name)
            return True, None
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def zip_integrity_check(data: bytes) -> Dict[str, Any]:
        """Comprehensive zip integrity check."""
        result = {
            'is_valid': False,
            'has_magic_number': data[:4] == b'PK\x03\x04' if len(data) >= 4 else False,
            'file_count': 0,
            'corrupted_files': [],
            'error': None
        }
        
        try:
            with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
                result['is_valid'] = True
                result['file_count'] = len(zf.namelist())
                
                bad_file = zf.testzip()
                if bad_file:
                    result['corrupted_files'].append(bad_file)
                    result['is_valid'] = False
        except Exception as e:
            result['error'] = str(e)
        
        return result


class ZipPredictorEvaluator:
    """
    Comprehensive evaluator for zip predictor model.
    
    Combines all metric types into a unified evaluation pipeline.
    """
    
    def __init__(self, content_type: str = 'auto'):
        """
        Initialize evaluator.
        
        Args:
            content_type: Type of content in zips ('auto', 'text', 'image', 'binary')
        """
        self.content_type = content_type
    
    def evaluate_single(
        self,
        pred: bytes,
        target: bytes
    ) -> EvaluationResult:
        """
        Evaluate a single prediction against target.
        
        Args:
            pred: Predicted zip bytes
            target: Ground truth zip bytes
            
        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult()
        
        # Byte-level metrics
        result.byte_accuracy = ByteLevelMetrics.accuracy(pred, target)
        result.byte_f1 = ByteLevelMetrics.byte_f1(pred, target)
        result.edit_distance = ByteLevelMetrics.normalized_edit_distance(pred, target)
        
        # Structure metrics
        pred_struct = StructureMetrics.parse_structure(pred)
        target_struct = StructureMetrics.parse_structure(target)
        
        result.file_count_accuracy = StructureMetrics.file_count_accuracy(pred_struct, target_struct)
        result.filename_similarity = StructureMetrics.filename_similarity(pred_struct, target_struct)
        result.structure_f1 = StructureMetrics.structure_f1(pred_struct, target_struct)
        
        # Content metrics
        result.content_hash_match, _ = ContentMetrics.content_hash_match(pred, target)
        result.content_similarity, file_sims = ContentMetrics.content_similarity(pred, target)
        result.per_file_metrics = {'similarity': file_sims}
        
        # Validity metrics
        result.valid_zip_rate = 1.0 if ValidityMetrics.is_valid_zip(pred) else 0.0
        success, _ = ValidityMetrics.decompression_test(pred)
        result.decompression_success_rate = 1.0 if success else 0.0
        
        return result
    
    def evaluate_batch(
        self,
        predictions: List[bytes],
        targets: List[bytes]
    ) -> EvaluationResult:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of predicted zip bytes
            targets: List of target zip bytes
            
        Returns:
            Aggregated EvaluationResult
        """
        if len(predictions) != len(targets):
            raise ValueError("Number of predictions must match number of targets")
        
        results = [
            self.evaluate_single(pred, target)
            for pred, target in zip(predictions, targets)
        ]
        
        # Aggregate results
        aggregated = EvaluationResult()
        
        for field_name in [
            'byte_accuracy', 'byte_f1', 'edit_distance',
            'file_count_accuracy', 'filename_similarity', 'structure_f1',
            'content_hash_match', 'content_similarity',
            'valid_zip_rate', 'decompression_success_rate'
        ]:
            values = [getattr(r, field_name) for r in results]
            setattr(aggregated, field_name, np.mean(values))
        
        return aggregated
    
    def evaluate_model(
        self,
        model,
        dataloader,
        device: str = 'cuda'
    ) -> EvaluationResult:
        """
        Evaluate model on a dataloader.
        
        Args:
            model: ZipPredictorModel instance
            dataloader: DataLoader with evaluation data
            device: Device to run on
            
        Returns:
            Aggregated evaluation results
        """
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_bytes = batch['target_bytes']
                
                # Generate predictions
                pred_bytes_list = model.generate_zip(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Convert targets to bytes
                for i, target in enumerate(target_bytes):
                    # Remove padding and special tokens
                    target_np = target.numpy()
                    target_np = target_np[target_np > 0]
                    all_targets.append(bytes(target_np.tolist()))
                
                all_preds.extend(pred_bytes_list)
        
        return self.evaluate_batch(all_preds, all_targets)


def evaluate_from_directories(
    pred_dir: Union[str, Path],
    target_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None
) -> EvaluationResult:
    """
    Evaluate predictions from directories of zip files.
    
    Args:
        pred_dir: Directory containing predicted zip files
        target_dir: Directory containing target zip files
        output_file: Optional file to save results
        
    Returns:
        EvaluationResult
    """
    pred_dir = Path(pred_dir)
    target_dir = Path(target_dir)
    
    evaluator = ZipPredictorEvaluator()
    
    predictions = []
    targets = []
    
    for target_file in sorted(target_dir.glob("*.zip")):
        pred_file = pred_dir / target_file.name
        if pred_file.exists():
            predictions.append(pred_file.read_bytes())
            targets.append(target_file.read_bytes())
    
    if not predictions:
        logger.warning("No matching prediction/target pairs found")
        return EvaluationResult()
    
    result = evaluator.evaluate_batch(predictions, targets)
    
    if output_file:
        output_file = Path(output_file)
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation results to {output_file}")
    
    return result
