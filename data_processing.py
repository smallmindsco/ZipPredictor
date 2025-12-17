"""
Data Processing Utilities

Handles:
- Zip file parsing and encoding
- Teacher model output processing  
- Dataset preparation
- Byte-level tokenization
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Optional, Dict, Any, List, Tuple, Iterator, Union
import io
import zipfile
import json
import struct
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Special tokens for byte-level encoding
SPECIAL_TOKENS = {
    'PAD': 0,
    'EOS': 1,
    'BOS': 2,
    'UNK': 3,
    'FILE_START': 4,
    'FILE_END': 5,
    'DIR_START': 6,
    'DIR_END': 7,
}

# Offset for regular bytes
BYTE_OFFSET = len(SPECIAL_TOKENS)


class ByteTokenizer:
    """
    Tokenizer for byte-level sequences with special tokens.
    
    Maps raw bytes to token indices and vice versa.
    """
    
    def __init__(self):
        self.special_tokens = SPECIAL_TOKENS
        self.byte_offset = BYTE_OFFSET
        self.vocab_size = 256 + BYTE_OFFSET
    
    def encode(
        self,
        data: bytes,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[int]:
        """Encode bytes to token indices."""
        tokens = []
        
        if add_bos:
            tokens.append(self.special_tokens['BOS'])
        
        for byte in data:
            tokens.append(byte + self.byte_offset)
        
        if add_eos:
            tokens.append(self.special_tokens['EOS'])
        
        return tokens
    
    def decode(
        self,
        tokens: List[int],
        skip_special: bool = True
    ) -> bytes:
        """Decode token indices to bytes."""
        byte_values = []
        
        for token in tokens:
            if token < self.byte_offset:
                if not skip_special:
                    # Handle special tokens
                    pass
            else:
                byte_values.append(token - self.byte_offset)
        
        return bytes(byte_values)
    
    def encode_zip_with_structure(
        self,
        zip_data: bytes
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Encode zip file with structural markers.
        
        Returns:
            Tuple of (tokens, metadata)
        """
        tokens = [self.special_tokens['BOS']]
        metadata = {'files': []}
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
                for name in zf.namelist():
                    info = zf.getinfo(name)
                    content = zf.read(name)
                    
                    # Add file start marker
                    tokens.append(self.special_tokens['FILE_START'])
                    
                    # Encode filename
                    for byte in name.encode('utf-8'):
                        tokens.append(byte + self.byte_offset)
                    
                    tokens.append(self.special_tokens['FILE_END'])
                    
                    # Encode content
                    for byte in content:
                        tokens.append(byte + self.byte_offset)
                    
                    tokens.append(self.special_tokens['FILE_END'])
                    
                    metadata['files'].append({
                        'name': name,
                        'size': len(content),
                        'compressed_size': info.compress_size
                    })
        except:
            # Fallback to raw encoding
            for byte in zip_data:
                tokens.append(byte + self.byte_offset)
        
        tokens.append(self.special_tokens['EOS'])
        
        return tokens, metadata


class ZipFileParser:
    """
    Parser for extracting structure and content from zip files.
    """
    
    @staticmethod
    def parse(zip_data: bytes) -> Dict[str, Any]:
        """
        Parse zip file into structured representation.
        
        Returns:
            Dictionary with structure:
            {
                'files': [
                    {
                        'path': str,
                        'content': bytes,
                        'size': int,
                        'compressed_size': int,
                        'compression': int,
                        'modified': datetime
                    }
                ],
                'raw_bytes': bytes
            }
        """
        result = {
            'files': [],
            'raw_bytes': zip_data
        }
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
                for name in zf.namelist():
                    info = zf.getinfo(name)
                    
                    file_info = {
                        'path': name,
                        'content': zf.read(name),
                        'size': info.file_size,
                        'compressed_size': info.compress_size,
                        'compression': info.compress_type,
                        'is_dir': info.is_dir()
                    }
                    
                    result['files'].append(file_info)
        except Exception as e:
            logger.warning(f"Failed to parse zip: {e}")
        
        return result
    
    @staticmethod
    def create(
        files: List[Dict[str, Any]],
        compression: int = zipfile.ZIP_DEFLATED,
        compresslevel: int = 6
    ) -> bytes:
        """
        Create zip file from structured representation.
        
        Args:
            files: List of {'path': str, 'content': bytes}
            compression: Compression type
            compresslevel: Compression level (1-9)
            
        Returns:
            Zip file bytes
        """
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', compression, compresslevel=compresslevel) as zf:
            for file_info in files:
                zf.writestr(file_info['path'], file_info['content'])
        
        return buffer.getvalue()


@dataclass
class TeacherOutputConfig:
    """Configuration for processing teacher model outputs."""
    output_type: str = 'image'  # 'image', 'text', 'audio', 'multimodal'
    image_format: str = 'png'
    text_encoding: str = 'utf-8'
    audio_format: str = 'wav'
    include_metadata: bool = True


class TeacherOutputProcessor:
    """
    Processes outputs from teacher model M into zip-ready format.
    
    Handles different output types (images, text, audio, etc.)
    and packages them appropriately.
    """
    
    def __init__(self, config: Optional[TeacherOutputConfig] = None):
        self.config = config or TeacherOutputConfig()
    
    def process(
        self,
        output: Any,
        prompt: str,
        additional_metadata: Optional[Dict] = None
    ) -> bytes:
        """
        Process teacher output into zip bytes.
        
        Args:
            output: Raw output from teacher model
            prompt: Original prompt
            additional_metadata: Extra metadata to include
            
        Returns:
            Zip file bytes
        """
        files = []
        
        # Determine output type and process
        if self.config.output_type == 'image':
            files.extend(self._process_image(output))
        elif self.config.output_type == 'text':
            files.extend(self._process_text(output))
        elif self.config.output_type == 'audio':
            files.extend(self._process_audio(output))
        elif self.config.output_type == 'multimodal':
            files.extend(self._process_multimodal(output))
        else:
            # Generic binary output
            files.append({
                'path': 'output.bin',
                'content': output if isinstance(output, bytes) else str(output).encode()
            })
        
        # Add metadata
        if self.config.include_metadata:
            metadata = {
                'prompt': prompt,
                'output_type': self.config.output_type,
                'num_files': len(files),
                **(additional_metadata or {})
            }
            files.append({
                'path': 'metadata.json',
                'content': json.dumps(metadata, indent=2).encode()
            })
        
        return ZipFileParser.create(files)
    
    def _process_image(self, output: Any) -> List[Dict[str, Any]]:
        """Process image output."""
        files = []
        
        # Handle different image representations
        if isinstance(output, bytes):
            files.append({
                'path': f'image.{self.config.image_format}',
                'content': output
            })
        elif isinstance(output, np.ndarray):
            # Convert numpy array to image bytes
            from PIL import Image
            img = Image.fromarray(output.astype('uint8'))
            buffer = io.BytesIO()
            img.save(buffer, format=self.config.image_format.upper())
            files.append({
                'path': f'image.{self.config.image_format}',
                'content': buffer.getvalue()
            })
        elif hasattr(output, 'images'):
            # Handle diffusers-style output
            for i, img in enumerate(output.images):
                buffer = io.BytesIO()
                img.save(buffer, format=self.config.image_format.upper())
                files.append({
                    'path': f'image_{i}.{self.config.image_format}',
                    'content': buffer.getvalue()
                })
        elif isinstance(output, (list, tuple)):
            # Multiple images
            for i, item in enumerate(output):
                files.extend([
                    {**f, 'path': f'{i}_{f["path"]}'}
                    for f in self._process_image(item)
                ])
        
        return files
    
    def _process_text(self, output: Any) -> List[Dict[str, Any]]:
        """Process text output."""
        if isinstance(output, str):
            content = output.encode(self.config.text_encoding)
        elif isinstance(output, bytes):
            content = output
        else:
            content = str(output).encode(self.config.text_encoding)
        
        return [{
            'path': 'output.txt',
            'content': content
        }]
    
    def _process_audio(self, output: Any) -> List[Dict[str, Any]]:
        """Process audio output."""
        files = []
        
        if isinstance(output, bytes):
            files.append({
                'path': f'audio.{self.config.audio_format}',
                'content': output
            })
        elif isinstance(output, np.ndarray):
            # Convert numpy to audio bytes
            import scipy.io.wavfile as wav
            buffer = io.BytesIO()
            wav.write(buffer, 44100, output)
            files.append({
                'path': 'audio.wav',
                'content': buffer.getvalue()
            })
        
        return files
    
    def _process_multimodal(self, output: Any) -> List[Dict[str, Any]]:
        """Process multimodal output."""
        files = []
        
        if isinstance(output, dict):
            for key, value in output.items():
                if 'image' in key.lower():
                    files.extend(self._process_image(value))
                elif 'text' in key.lower():
                    files.extend(self._process_text(value))
                elif 'audio' in key.lower():
                    files.extend(self._process_audio(value))
        
        return files


class PromptZipDataset(Dataset):
    """
    Dataset of (prompt, zip_bytes) pairs for training.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer: Optional[ByteTokenizer] = None,
        max_prompt_len: int = 512,
        max_output_len: int = 65536,
        text_tokenizer: Any = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing prompt.txt and output.zip pairs
            tokenizer: ByteTokenizer for encoding
            max_prompt_len: Maximum prompt length
            max_output_len: Maximum output length
            text_tokenizer: Tokenizer for text prompts
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer or ByteTokenizer()
        self.max_prompt_len = max_prompt_len
        self.max_output_len = max_output_len
        self.text_tokenizer = text_tokenizer
        
        # Find all data pairs
        self.samples = self._find_samples()
        logger.info(f"Found {len(self.samples)} samples in {data_dir}")
    
    def _find_samples(self) -> List[Dict[str, Path]]:
        """Find all prompt/output pairs in data directory."""
        samples = []
        
        # Look for pairs: prompt_X.txt and output_X.zip
        for prompt_file in self.data_dir.glob("*.txt"):
            stem = prompt_file.stem
            if stem.startswith("prompt_"):
                idx = stem.replace("prompt_", "")
                output_file = self.data_dir / f"output_{idx}.zip"
                if output_file.exists():
                    samples.append({
                        'prompt': prompt_file,
                        'output': output_file
                    })
        
        # Also look for JSON index
        index_file = self.data_dir / "index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            for item in index:
                samples.append({
                    'prompt': self.data_dir / item['prompt'],
                    'output': self.data_dir / item['output']
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load prompt
        prompt = sample['prompt'].read_text().strip()
        
        # Load zip bytes
        zip_bytes = sample['output'].read_bytes()
        
        # Encode prompt
        if self.text_tokenizer is not None:
            encoded = self.text_tokenizer(
                prompt,
                max_length=self.max_prompt_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            # Character-level fallback
            chars = [ord(c) for c in prompt[:self.max_prompt_len]]
            input_ids = torch.zeros(self.max_prompt_len, dtype=torch.long)
            input_ids[:len(chars)] = torch.tensor(chars)
            attention_mask = torch.zeros(self.max_prompt_len, dtype=torch.long)
            attention_mask[:len(chars)] = 1
        
        # Encode zip bytes
        target_tokens = self.tokenizer.encode(zip_bytes[:self.max_output_len])
        target_bytes = torch.zeros(self.max_output_len, dtype=torch.long)
        target_bytes[:len(target_tokens)] = torch.tensor(target_tokens)
        target_mask = torch.zeros(self.max_output_len, dtype=torch.long)
        target_mask[:len(target_tokens)] = 1
        
        # Parse zip structure
        zip_info = ZipFileParser.parse(zip_bytes)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_bytes': target_bytes,
            'target_mask': target_mask,
            'file_count': torch.tensor(len(zip_info['files']), dtype=torch.long),
            'original_length': torch.tensor(len(zip_bytes), dtype=torch.long)
        }


class StreamingTeacherDataset(IterableDataset):
    """
    Streaming dataset that generates training data on-the-fly
    from teacher model outputs.
    """
    
    def __init__(
        self,
        teacher_model: Any,
        prompts_source: Union[Path, List[str], Iterator[str]],
        output_processor: Optional[TeacherOutputProcessor] = None,
        tokenizer: Optional[ByteTokenizer] = None,
        max_prompt_len: int = 512,
        max_output_len: int = 65536,
        text_tokenizer: Any = None,
        cache_dir: Optional[Path] = None
    ):
        self.teacher_model = teacher_model
        self.output_processor = output_processor or TeacherOutputProcessor()
        self.tokenizer = tokenizer or ByteTokenizer()
        self.max_prompt_len = max_prompt_len
        self.max_output_len = max_output_len
        self.text_tokenizer = text_tokenizer
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Setup prompts source
        if isinstance(prompts_source, Path):
            self.prompts = self._load_prompts_from_file(prompts_source)
        elif isinstance(prompts_source, list):
            self.prompts = prompts_source
        else:
            self.prompts = prompts_source
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_prompts_from_file(self, path: Path) -> List[str]:
        """Load prompts from file."""
        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
            return [item if isinstance(item, str) else item['prompt'] for item in data]
        else:
            return path.read_text().strip().split('\n')
    
    def _get_cache_key(self, prompt: str) -> str:
        """Get cache key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over dataset, generating from teacher as needed."""
        for prompt in self.prompts:
            # Check cache
            if self.cache_dir:
                cache_key = self._get_cache_key(prompt)
                cache_path = self.cache_dir / f"{cache_key}.zip"
                
                if cache_path.exists():
                    zip_bytes = cache_path.read_bytes()
                else:
                    # Generate from teacher
                    output = self.teacher_model.generate(prompt)
                    zip_bytes = self.output_processor.process(output, prompt)
                    cache_path.write_bytes(zip_bytes)
            else:
                output = self.teacher_model.generate(prompt)
                zip_bytes = self.output_processor.process(output, prompt)
            
            # Encode and yield
            yield self._encode_sample(prompt, zip_bytes)
    
    def _encode_sample(
        self,
        prompt: str,
        zip_bytes: bytes
    ) -> Dict[str, torch.Tensor]:
        """Encode a single sample."""
        # Encode prompt
        if self.text_tokenizer is not None:
            encoded = self.text_tokenizer(
                prompt,
                max_length=self.max_prompt_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            chars = [ord(c) for c in prompt[:self.max_prompt_len]]
            input_ids = torch.zeros(self.max_prompt_len, dtype=torch.long)
            input_ids[:len(chars)] = torch.tensor(chars)
            attention_mask = torch.zeros(self.max_prompt_len, dtype=torch.long)
            attention_mask[:len(chars)] = 1
        
        # Encode zip bytes
        target_tokens = self.tokenizer.encode(zip_bytes[:self.max_output_len])
        target_bytes = torch.zeros(self.max_output_len, dtype=torch.long)
        target_bytes[:len(target_tokens)] = torch.tensor(target_tokens)
        target_mask = torch.zeros(self.max_output_len, dtype=torch.long)
        target_mask[:len(target_tokens)] = 1
        
        zip_info = ZipFileParser.parse(zip_bytes)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_bytes': target_bytes,
            'target_mask': target_mask,
            'file_count': torch.tensor(len(zip_info['files']), dtype=torch.long),
            'original_length': torch.tensor(len(zip_bytes), dtype=torch.long)
        }


def prepare_dataset_from_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    teacher_model: Any,
    output_processor: Optional[TeacherOutputProcessor] = None,
    num_samples: Optional[int] = None
):
    """
    Prepare a dataset by generating outputs from teacher model.
    
    Args:
        input_dir: Directory containing prompt files
        output_dir: Directory to save processed dataset
        teacher_model: Teacher model for generation
        output_processor: Processor for teacher outputs
        num_samples: Maximum number of samples to generate
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = output_processor or TeacherOutputProcessor()
    
    # Find prompt files
    prompt_files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.json"))
    
    if num_samples:
        prompt_files = prompt_files[:num_samples]
    
    index = []
    
    for i, prompt_file in enumerate(prompt_files):
        logger.info(f"Processing {i+1}/{len(prompt_files)}: {prompt_file.name}")
        
        # Load prompt
        if prompt_file.suffix == '.json':
            with open(prompt_file) as f:
                prompts = json.load(f)
        else:
            prompts = [prompt_file.read_text().strip()]
        
        for j, prompt in enumerate(prompts):
            # Generate
            output = teacher_model.generate(prompt)
            
            # Process to zip
            zip_bytes = processor.process(output, prompt)
            
            # Save
            idx = f"{i}_{j}"
            prompt_path = output_dir / f"prompt_{idx}.txt"
            output_path = output_dir / f"output_{idx}.zip"
            
            prompt_path.write_text(prompt)
            output_path.write_bytes(zip_bytes)
            
            index.append({
                'prompt': f"prompt_{idx}.txt",
                'output': f"output_{idx}.zip"
            })
    
    # Save index
    with open(output_dir / "index.json", 'w') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Created dataset with {len(index)} samples in {output_dir}")
