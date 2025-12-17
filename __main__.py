#!/usr/bin/env python
"""
Command-line interface for Zip Predictor.

Usage:
    # Train from safetensors
    python -m zip_predictor train --teacher model.safetensors --prompts prompts.txt --output ./checkpoints
    
    # Generate from trained model
    python -m zip_predictor generate --model ./checkpoints/best --prompt "Generate an image" --output result.zip
    
    # Evaluate predictions
    python -m zip_predictor evaluate --predictions ./preds --targets ./targets --output results.json
"""

import argparse
import sys
import json
from pathlib import Path
import torch


def train_command(args):
    """Handle train command."""
    from zip_predictor import train_from_safetensors, TrainingConfig
    
    # Load prompts
    prompts_path = Path(args.prompts)
    if prompts_path.suffix == '.json':
        with open(prompts_path) as f:
            prompts = json.load(f)
        if isinstance(prompts[0], dict):
            prompts = [p['prompt'] for p in prompts]
    else:
        prompts = prompts_path.read_text().strip().split('\n')
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Training config
    config_overrides = {}
    if args.batch_size:
        config_overrides['batch_size'] = args.batch_size
    if args.learning_rate:
        config_overrides['learning_rate'] = args.learning_rate
    if args.max_steps:
        config_overrides['max_steps'] = args.max_steps
    if args.wandb:
        config_overrides['use_wandb'] = True
        config_overrides['wandb_project'] = args.wandb_project or 'zip-predictor'
    
    # Train
    model = train_from_safetensors(
        safetensors_path=args.teacher,
        prompts=prompts,
        model_size=args.model_size,
        output_dir=args.output,
        **config_overrides
    )
    
    print(f"Training complete! Model saved to {args.output}")


def generate_command(args):
    """Handle generate command."""
    from zip_predictor import load_generator
    
    # Load generator
    generator = load_generator(
        args.model,
        device=args.device
    )
    
    # Get prompt(s)
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        prompts_path = Path(args.prompts_file)
        if prompts_path.suffix == '.json':
            with open(prompts_path) as f:
                prompts = json.load(f)
        else:
            prompts = prompts_path.read_text().strip().split('\n')
    else:
        print("Error: Must provide --prompt or --prompts-file")
        sys.exit(1)
    
    # Generate
    output_dir = Path(args.output) if len(prompts) > 1 else None
    
    if len(prompts) == 1:
        result = generator(
            prompts[0],
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Save
        output_path = Path(args.output)
        if args.extract:
            output_path.mkdir(parents=True, exist_ok=True)
            import zipfile
            import io
            with zipfile.ZipFile(io.BytesIO(result), 'r') as zf:
                zf.extractall(output_path)
            print(f"Extracted to {output_path}")
        else:
            output_path.write_bytes(result)
            print(f"Saved to {output_path}")
    else:
        results = generator(
            prompts,
            output_dir=args.output,
            extract=args.extract,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"Generated {len(results)} zip files to {args.output}")


def evaluate_command(args):
    """Handle evaluate command."""
    from zip_predictor import evaluate_from_directories
    
    result = evaluate_from_directories(
        pred_dir=args.predictions,
        target_dir=args.targets,
        output_file=args.output
    )
    
    print(result.summary())


def info_command(args):
    """Handle info command."""
    from zip_predictor import ZipPredictorModel, create_model
    import safetensors.torch as st
    
    if args.model:
        model_path = Path(args.model)
        
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            print("Model Configuration:")
            print(json.dumps(config, indent=2))
        
        # Load weights and show stats
        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            state_dict = st.load_file(str(weights_path))
            
            print(f"\nModel Statistics:")
            print(f"  Total parameters: {sum(t.numel() for t in state_dict.values()):,}")
            print(f"  Number of tensors: {len(state_dict)}")
            print(f"  File size: {weights_path.stat().st_size / 1e6:.2f} MB")
    
    if args.safetensors:
        safetensors_path = Path(args.safetensors)
        state_dict = st.load_file(str(safetensors_path))
        
        print(f"\nSafetensors File: {safetensors_path}")
        print(f"  Total parameters: {sum(t.numel() for t in state_dict.values()):,}")
        print(f"  Number of tensors: {len(state_dict)}")
        print(f"  File size: {safetensors_path.stat().st_size / 1e6:.2f} MB")
        
        print("\n  Layer shapes:")
        for name, tensor in list(state_dict.items())[:20]:
            print(f"    {name}: {list(tensor.shape)}")
        if len(state_dict) > 20:
            print(f"    ... and {len(state_dict) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Zip Predictor: Train models to predict zip file outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train zip predictor model')
    train_parser.add_argument('--teacher', required=True, help='Path to teacher model safetensors')
    train_parser.add_argument('--prompts', required=True, help='Path to prompts file (txt or json)')
    train_parser.add_argument('--output', default='./checkpoints', help='Output directory')
    train_parser.add_argument('--model-size', choices=['base', 'large', 'xl'], default='base')
    train_parser.add_argument('--batch-size', type=int, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--max-steps', type=int, help='Maximum training steps')
    train_parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    train_parser.add_argument('--wandb-project', help='W&B project name')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate zip files from prompts')
    gen_parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    gen_parser.add_argument('--prompt', help='Single prompt to generate from')
    gen_parser.add_argument('--prompts-file', help='File with multiple prompts')
    gen_parser.add_argument('--output', required=True, help='Output path (file or directory)')
    gen_parser.add_argument('--extract', action='store_true', help='Extract zip contents')
    gen_parser.add_argument('--device', default='auto', help='Device (auto, cuda, cpu)')
    gen_parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    gen_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    gen_parser.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate predictions')
    eval_parser.add_argument('--predictions', required=True, help='Directory with predicted zips')
    eval_parser.add_argument('--targets', required=True, help='Directory with target zips')
    eval_parser.add_argument('--output', help='Output JSON file for results')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model/file information')
    info_parser.add_argument('--model', help='Path to model checkpoint')
    info_parser.add_argument('--safetensors', help='Path to safetensors file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'info':
        info_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
