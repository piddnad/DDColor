import argparse
import cv2
import os
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from inference.colorization_pipeline import ImageColorizationPipeline


class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


class ImageColorizationPipelineHF(ImageColorizationPipeline):
    def __init__(self, model, input_size):
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ddcolor_modelscope")
    parser.add_argument(
        "--input",
        type=str,
        default="figure/",
        help="input test image folder or video path",
    )
    parser.add_argument(
        "--output", type=str, default="results", help="output folder or video path"
    )
    parser.add_argument(
        "--input_size", type=int, default=512, help="input size for model"
    )

    args = parser.parse_args()

    ddcolor_model = DDColorHF.from_pretrained(f"piddnad/{args.model_name}")

    print(f"Output path: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    colorizer = ImageColorizationPipelineHF(
        model=ddcolor_model, input_size=args.input_size
    )

    for name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input, name))
        image_out = colorizer.process(img)
        cv2.imwrite(os.path.join(args.output, name), image_out)


if __name__ == "__main__":
    main()
