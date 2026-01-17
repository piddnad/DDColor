#!/usr/bin/env python
"""
Export DDColor model to ONNX format.

Usage:
    python scripts/export_onnx.py --model_path pretrain/ddcolor_paper_tiny.pth --export_path weights/ddcolor-tiny.onnx
"""

import os
import sys
import argparse

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import onnx
import onnxsim

from basicsr.archs.ddcolor_arch import DDColor

from onnx import load_model, save_model, shape_inference
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


def parse_args():
    parser = argparse.ArgumentParser(description="Export DDColor model to ONNX.")
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Input image dimension.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Input batch size.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model weights (.pt file).",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["tiny", "large"],
        help="DDColor model size.",
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="MultiScaleColorDecoder",
        choices=["MultiScaleColorDecoder", "SingleColorDecoder"],
        help="Decoder type.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="./model.onnx",
        help="Path to export ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version.",
    )

    return parser.parse_args()


def create_onnx_export(args):
    input_size = args.input_size
    device = torch.device('cpu')
    
    encoder_name = 'convnext-t' if args.model_size == 'tiny' else 'convnext-l'

    if args.decoder_type == 'MultiScaleColorDecoder':
        model = DDColor(
            encoder_name=encoder_name,
            decoder_name='MultiScaleColorDecoder',
            input_size=[input_size, input_size],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        ).to(device)
    elif args.decoder_type == 'SingleColorDecoder':
        model = DDColor(
            encoder_name=encoder_name,
            decoder_name='SingleColorDecoder',
            input_size=[input_size, input_size],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=256,
        ).to(device)
    else:
        raise ValueError(f"decoder_type not implemented: {args.decoder_type}")

    model.load_state_dict(
        torch.load(args.model_path, map_location=device)['params'],
        strict=False)
    model.eval()

    channels = 3  # RGB image has 3 channels

    random_input = torch.rand((args.batch_size, channels, input_size, input_size), dtype=torch.float32)

    dynamic_axes = {}
    if args.batch_size == 0:
        dynamic_axes[0] = "batch"
    if input_size == 0:
        dynamic_axes[2] = "height"
        dynamic_axes[3] = "width"
    
    # Create output directory if needed
    export_dir = os.path.dirname(args.export_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
    
    torch.onnx.export(
        model,
        random_input,
        args.export_path,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": dynamic_axes,
            "output": dynamic_axes
        },
    )


def check_onnx_export(export_path):
    save_model(
        shape_inference.infer_shapes(
            load_model(export_path),
            check_type=True,
            strict_mode=True,
            data_prop=True
        ),
        export_path
    )

    save_model(
        SymbolicShapeInference.infer_shapes(
            load_model(export_path),
            auto_merge=True,
            guess_output_rank=True
        ),
        export_path,
    )

    model_onnx = onnx.load(export_path)
    onnx.checker.check_model(model_onnx)

    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, export_path)


if __name__ == '__main__':
    args = parse_args()

    create_onnx_export(args)
    print(f'ONNX file successfully created at {args.export_path}')
    
    check_onnx_export(args.export_path)
    print(f'ONNX file at {args.export_path} verified shapes and simplified')
