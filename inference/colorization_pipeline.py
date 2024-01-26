import argparse
import cv2
import numpy as np
import os
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
    def process(self, img):
        self.height, self.width = img.shape[:2]
        # print(self.width, self.height)
        # if self.width * self.height < 100000:
        #     self.input_size = 256

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
    parser.add_argument('--model_path', type=str, default='pretrain/net_g_200000.pth')
    parser.add_argument('--input', type=str, default='figure/', help='input test image folder or video path')
    parser.add_argument('--output', type=str, default='results', help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=512, help='input size for model')
    parser.add_argument('--model_size', type=str, default='large', help='ddcolor model size')
    args = parser.parse_args()

    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)
    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    colorizer = ImageColorizationPipeline(model_path=args.model_path, input_size=args.input_size, model_size=args.model_size)

    for name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input, name))
        image_out = colorizer.process(img)
        cv2.imwrite(os.path.join(args.output, name), image_out)


if __name__ == '__main__':
    main()
