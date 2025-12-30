"""
Evaluate trained model
"""
import torch
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mamba_rna import create_model
from utils.data_loader import RNATokenizer, load_rnagym_data
from utils.metrics import calculate_all_metrics, print_metrics
from utils.visualization import plot_prediction_scatter, save_results_summary
import numpy as np
from tqdm import tqdm


def evaluate_model(model_path, data_dir, dataset_names, output_dir):
    """Evaluate trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model_config = checkpoint['model_config']
    
    # Create model
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\nLoaded model: {model_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint['val_loss']:.6f}")
    
    # Load data
    tokenizer = RNATokenizer()
    train_loader, val_loader, dataset = load_rnagym_data(
        data_dir=data_dir,
        dataset_names=dataset_names,
        tokenizer=tokenizer,
        batch_size=32,
        train_ratio=0.8,
        max_length=512,
        num_workers=0,
        normalize=True
    )
    
    # Evaluate
    print("\nStarting evaluation...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Evaluating')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['fitness'].to(device)
            
            predictions = model(input_ids)
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Denormalize
    if dataset.normalize:
        all_preds_original = dataset.denormalize(all_preds)
        all_targets_original = dataset.denormalize(all_targets)
    else:
        all_preds_original = all_preds
        all_targets_original = all_targets
    
    # Calculate metrics
    metrics = calculate_all_metrics(all_targets_original, all_preds_original)
    
    print("\nFinal Evaluation Results:")
    print_metrics(metrics)
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    scatter_path = os.path.join(output_dir, 'prediction_scatter.png')
    plot_prediction_scatter(all_targets_original, all_preds_original, metrics, save_path=scatter_path)
    
    # Save results summary
    summary_path = os.path.join(output_dir, 'results_summary.txt')
    save_results_summary(metrics, save_path=summary_path)
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', type=str, default='outputs_test/best_model.pt',
                        help='Model file path')
    parser.add_argument('--data_dir', type=str, default='data/RNAGym',
                        help='Data directory')
    parser.add_argument('--datasets', type=str, nargs='+', default=['Andreasson_2020_ribozyme'],
                        help='Dataset name(s) (can specify multiple)')
    parser.add_argument('--output_dir', type=str, default='outputs_test',
                        help='Output directory')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.datasets, args.output_dir)
