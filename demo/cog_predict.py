# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import cv2
import torch
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Add project root to path for imports
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_large = build_ddcolor_model(
            DDColor,
            model_path="checkpoints/ddcolor_modelscope.pth",
            input_size=512,
            model_size="large",
            device=device,
        )
        model_tiny = build_ddcolor_model(
            DDColor,
            model_path="checkpoints/ddcolor_paper_tiny.pth",
            input_size=512,
            model_size="tiny",
            device=device,
        )
        self.colorizer = ColorizationPipeline(model_large, input_size=512, device=device)
        self.colorizer_tiny = ColorizationPipeline(model_tiny, input_size=512, device=device)

    def predict(
        self,
        image: Path = Input(description="Grayscale input image."),
        model_size: str = Input(
            description="Choose the model size.",
            choices=["large", "tiny"],
            default="large",
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Failed to read image: {image}")
        colorizer = self.colorizer_tiny if model_size == "tiny" else self.colorizer
        image_out = colorizer.process(img)
        out_path = "/tmp/out.png"
        cv2.imwrite(out_path, image_out)
        return Path(out_path)
