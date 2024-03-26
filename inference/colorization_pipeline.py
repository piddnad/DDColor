import argparse
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F

class ImageColorizationPipeline(object):

    def __init__(self, model_path, input_size=256, model_size='large'):
        # Initialize the colorization pipeline
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.decoder_type = "MultiScaleColorDecoder"

        # Initialize the colorization model
        if self.decoder_type == 'MultiScaleColorDecoder':
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='MultiScaleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ).to(self.device)
        else:
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='SingleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=256,
            ).to(self.device)

        # Load pre-trained weights for the model
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()

    @torch.no_grad()
    def process(self, img, saturation_reduction=0.4, gamma=1.0, brightness=0, contrast=1.0):
        # Preprocess input image and perform colorization
        
        # Check if the input image is already grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:
            if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 0], img[:, :, 2]):
                print("Input image is already grayscale.")
            else:
                # Convert input image to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                print("Converted input image to grayscale.")

        # Convert image to float32 and extract luminance (L) channel
        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

        # Save original image dimensions
        orig_height, orig_width = img.shape[:2]

        # Resize image and create grayscale Lab representation
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        # Prepare input tensor for the model
        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        
        # Generate colorized output
        output_ab = self.model(tensor_gray_rgb).cpu()
        output_ab_resize = F.interpolate(output_ab, size=(img.shape[0], img.shape[1]))[0].float().numpy().transpose(1, 2, 0)

        # Resize original luminance (L) channel to match colorized image size
        orig_l_resized = cv2.resize(orig_l, (output_ab_resize.shape[1], output_ab_resize.shape[0]))

        # Ensure orig_l_resized has 3 dimensions
        if len(orig_l_resized.shape) == 2:
            orig_l_resized = np.expand_dims(orig_l_resized, axis=-1)

        # Concatenate resized luminance (L) channel with colorized image
        output_lab = np.concatenate((orig_l_resized, output_ab_resize), axis=-1)

        # Convert output to uint8
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        output_img = (output_bgr * 255.0).round().astype(np.uint8)

        # Resize output to original dimensions
        output_img = cv2.resize(output_img, (orig_width, orig_height))

        # Apply color adjustments
        output_img = self.adjust_colors(output_img, gamma=gamma, brightness=brightness, contrast=contrast)

        # Apply saturation reduction
        output_img = self.desaturate(output_img, saturation_reduction)
        
        return output_img

    @staticmethod
    def desaturate(img, saturation_reduction=0.4):
        # Desaturate the image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = hsv[..., 1] * (1 - saturation_reduction)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_colors(img, gamma=1.0, brightness=0, contrast=1.0):
        # Adjust gamma, brightness, and contrast
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.clip(((img / 255.0) ** gamma) * 255.0 * contrast + brightness, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrain/net_g_200000.pth')
    parser.add_argument('--input', type=str, default='figure/', help='input test image folder or video path')
    parser.add_argument('--output', type=str, default='results', help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=512, help='input size for model')
    parser.add_argument('--model_size', type=str, default='large', help='ddcolor model size')
    parser.add_argument('--saturation_reduction', type=float, default=0.4, help='saturation reduction factor (default: 0.4)')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma correction factor (default: 1.0)')
    parser.add_argument('--brightness', type=int, default=0, help='brightness adjustment (default: 0)')
    parser.add_argument('--contrast', type=float, default=1.0, help='contrast adjustment factor (default: 1.0)')
    parser.add_argument('--auto_correct_color', action='store_true', help='perform automatic color correction')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)
    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    # Initialize colorization pipeline
    colorizer = ImageColorizationPipeline(model_path=args.model_path, input_size=args.input_size, model_size=args.model_size)

    # Process each image in the input directory
    for name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input, name))

        if args.auto_correct_color:
            # Perform automatic color correction
            img = colorizer.auto_correct_color_levels(img)

        # Perform colorization
        image_out = colorizer.process(img, saturation_reduction=args.saturation_reduction)
        cv2.imwrite(os.path.join(args.output, name), image_out)

if __name__ == '__main__':
    main()import argparse
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F

class ImageColorizationPipeline(object):

    def __init__(self, model_path, input_size=256, model_size='large'):
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.decoder_type = "MultiScaleColorDecoder"

        if self.decoder_type == 'MultiScaleColorDecoder':
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='MultiScaleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ).to(self.device)
        else:
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='SingleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=256,
            ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()

    @torch.no_grad()
    def process(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image '{img_path}'. Skipping...")
            return None

        self.height, self.width = img.shape[:2]
        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # resize rgb image -> lab -> get grey -> rgb
        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor_gray_rgb).cpu()  # (1, 2, self.height, self.width)

        # resize ab -> concat original l -> rgb
        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)
        return output_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrain/ddcolor_modelscope.pth')
    parser.add_argument('--input', type=str, default='assets/test_images/', help='input test image folder or video path')
    parser.add_argument('--output', type=str, default='assets/test_images/output/', help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=512, help='input size for model')
    parser.add_argument('--model_size', type=str, default='large', help='ddcolor model size')
    args = parser.parse_args()

    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)
    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    colorizer = ImageColorizationPipeline(model_path=args.model_path, input_size=args.input_size, model_size=args.model_size)

    for name in tqdm(img_list):
        img_path = os.path.join(args.input, name)
        output_img = colorizer.process(img_path)
        if output_img is not None:
            cv2.imwrite(os.path.join(args.output, name), output_img)

if __name__ == '__main__':
    main()
