# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cv2
import numpy as np
from subprocess import call
import torch
import torch.nn.functional as F
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # download the weights to "checkpoints"

        from basicsr.archs.ddcolor_arch import DDColor

        class ImageColorizationPipeline(object):
            def __init__(self, model_path, input_size=256, model_size="large"):
                self.input_size = input_size
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")

                if model_size == "tiny":
                    self.encoder_name = "convnext-t"
                else:
                    self.encoder_name = "convnext-l"

                self.decoder_type = "MultiScaleColorDecoder"

                self.model = DDColor(
                    encoder_name=self.encoder_name,
                    decoder_name="MultiScaleColorDecoder",
                    input_size=[self.input_size, self.input_size],
                    num_output_channels=2,
                    last_norm="Spectral",
                    do_normalize=False,
                    num_queries=100,
                    num_scales=3,
                    dec_layers=9,
                ).to(self.device)

                self.model.load_state_dict(
                    torch.load(model_path, map_location=torch.device("cpu"))["params"],
                    strict=False,
                )
                self.model.eval()

            @torch.no_grad()
            def process(self, img):
                self.height, self.width = img.shape[:2]
                img = (img / 255.0).astype(np.float32)
                orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

                # resize rgb image -> lab -> get grey -> rgb
                img = cv2.resize(img, (self.input_size, self.input_size))
                img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
                img_gray_lab = np.concatenate(
                    (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1
                )
                img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

                tensor_gray_rgb = (
                    torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                output_ab = self.model(
                    tensor_gray_rgb
                ).cpu()  # (1, 2, self.height, self.width)

                # resize ab -> concat original l -> rgb
                output_ab_resize = (
                    F.interpolate(output_ab, size=(self.height, self.width))[0]
                    .float()
                    .numpy()
                    .transpose(1, 2, 0)
                )
                output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
                output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

                output_img = (output_bgr * 255.0).round().astype(np.uint8)

                return output_img

        self.colorizer = ImageColorizationPipeline(
            model_path="checkpoints/ddcolor_modelscope.pth",
            input_size=512,
            model_size="large",
        )
        self.colorizer_tiny = ImageColorizationPipeline(
            model_path="checkpoints/ddcolor_paper_tiny.pth",
            input_size=512,
            model_size="tiny",
        )

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
        colorizer = self.colorizer_tiny if model_size == "tiny" else self.colorizer
        image_out = colorizer.process(img)
        out_path = "/tmp/out.png"
        cv2.imwrite(out_path, image_out)
        return Path(out_path)
