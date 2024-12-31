import os
import cv2
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from ddcolor_model import DDColor


class ImageColorizationPipeline:
    def __init__(self, model_path, input_size=256, model_size='large'):
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder_name = 'convnext-t' if model_size == 'tiny' else 'convnext-l'
        self.decoder_type = 'MultiScaleColorDecoder'

        self.model = DDColor(
            encoder_name=self.encoder_name,
            decoder_name=self.decoder_type,
            input_size=[self.input_size, self.input_size],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu')['params'],
            strict=False
        )
        self.model.eval()

    @torch.no_grad()
    def process(self, img):
        height, width = img.shape[:2]

        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # Resize and convert image to grayscale
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor_gray_rgb).cpu()  # (1, 2, self.input_size, self.input_size)

        # Resize output and concatenate with original L channel
        output_ab_resized = F.interpolate(output_ab, size=(height, width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)
        return output_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt', help='Path to the model weights')
    parser.add_argument('--input', type=str, default='assets/test_images', help='Input image folder')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--input_size', type=int, default=512, help='Input size for the model')
    parser.add_argument('--model_size', type=str, default='large', help='DDColor model size (tiny or large)')
    args = parser.parse_args()

    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)
    file_list = os.listdir(args.input)
    assert len(file_list) > 0, "No images found in the input directory."

    colorizer = ImageColorizationPipeline(model_path=args.model_path, input_size=args.input_size, model_size=args.model_size)

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
