import torch
from basicsr.archs.ddcolor_arch import DDColor

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from .colorization_pipeline import ImageColorizationPipeline


class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


class ImageColorizationPipelineHF(ImageColorizationPipeline):
    def __init__(self, model):
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = model.to(self.device)
        self.model.eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ddcolor_modelscope')
    args = parser.parse_args()

    ddcolor_model = DDColorHF.from_pretrained(f"piddnad/{args.model_name}")

    print(f'Output path: {args.output}')
    os.makedirs(args.output, exist_ok=True)
    img_list = os.listdir(args.input)
    assert len(img_list) > 0

    colorizer = ImageColorizationPipelineHF(model=ddcolor_model)

    for name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input, name))
        image_out = colorizer.process(img)
        cv2.imwrite(os.path.join(args.output, name), image_out)


if __name__ == '__main__':
    main()
