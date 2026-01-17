#!/usr/bin/env python
"""
DDColor inference script.

Supports two modes:
1. Local weights: python scripts/infer.py --model_path path/to/model.pt --input ./images
2. Hugging Face:  python scripts/infer.py --model_name ddcolor_modelscope --input ./images
"""

import os
import sys
import argparse

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import torch
from tqdm import tqdm

from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model


def main():
    parser = argparse.ArgumentParser(description="DDColor inference script")
    
    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model_path', type=str,
        help='Path to the local model weights (.pt file)'
    )
    model_group.add_argument(
        '--model_name', type=str,
        help='Hugging Face model name (e.g., ddcolor_modelscope, ddcolor_paper, ddcolor_artistic, ddcolor_paper_tiny)'
    )
    
    # Common arguments
    parser.add_argument('--input', type=str, default='assets/test_images', help='Input image folder')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--input_size', type=int, default=512, help='Input size for the model')
    parser.add_argument('--model_size', type=str, default='large', choices=['tiny', 'large'],
                        help='DDColor model size (only used with --model_path)')
    
    args = parser.parse_args()

    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)
    
    file_list = os.listdir(args.input)
    assert len(file_list) > 0, "No images found in the input directory."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_path:
        # Local weights mode
        model = build_ddcolor_model(
            DDColor,
            model_path=args.model_path,
            input_size=args.input_size,
            model_size=args.model_size,
            device=device,
        )
    else:
        # Hugging Face mode
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        model_name = args.model_name
        if not os.path.isdir(model_name):
            model_name = f"piddnad/{model_name}"
        
        model = DDColorHF.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

    colorizer = ColorizationPipeline(model, input_size=args.input_size, device=device)

    for file_name in tqdm(file_list):
        img_path = os.path.join(args.input, file_name)
        img = cv2.imread(img_path)
        if img is not None:
            image_out = colorizer.process(img)
            cv2.imwrite(os.path.join(args.output, file_name), image_out)
        else:
            print(f"Failed to read {img_path}")


if __name__ == '__main__':
    main()
