#!/usr/bin/env python3
"""
Diagnostic script to inspect the saved model structure
"""

import torch
import json

# Load the checkpoint to inspect its structure
MODEL_PATH = "./models/model_hybrid_50e.pt"


def inspect_checkpoint():
    """Inspect the saved model structure"""
    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location='cpu',
            weights_only=True)

        print("=== Saved Model Structure ===")
        if isinstance(checkpoint, dict):
            print(
                "Checkpoint is a dictionary with keys:",
                list(
                    checkpoint.keys()))
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        print("\n=== Layer Keys and Shapes ===")
        for key, value in state_dict.items():
            if hasattr(value, 'shape'):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {type(value)}")

        # Count classifier layers
        classifier_keys = [
            k for k in state_dict.keys() if k.startswith('classifier.')]
        print(f"\n=== Classifier Layers Found: {len(classifier_keys)} ===")
        for key in sorted(classifier_keys):
            if hasattr(state_dict[key], 'shape'):
                print(f"  {key}: {state_dict[key].shape}")

        return state_dict

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


if __name__ == "__main__":
    inspect_checkpoint()
