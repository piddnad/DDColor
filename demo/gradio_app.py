import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import torch
import uuid

import gradio as gr
from gradio_imageslider import ImageSlider

from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model


model_path = 'modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt'
input_size = 512
model_size = 'large'


# Initialize
_model = build_ddcolor_model(
    DDColor,
    model_path=model_path,
    input_size=input_size,
    model_size=model_size,
    decoder_type="MultiScaleColorDecoder",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
colorizer = ColorizationPipeline(_model, input_size=input_size)


# Create inference function for gradio app
def colorize(img):
    image_out = colorizer.process(img)
    # Generate a unique filename using UUID
    unique_imgfilename = str(uuid.uuid4()) + '.png'
    cv2.imwrite(unique_imgfilename, image_out)
    return (img, unique_imgfilename)


# Gradio demo using the Image-Slider custom component
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            bw_image = gr.Image(label='Black and White Input Image')
            btn = gr.Button('Convert using DDColor')
        with gr.Column():
            col_image_slider = ImageSlider(position=0.5,
                                           label='Colored Image with Slider-view')

    btn.click(colorize, bw_image, col_image_slider)


if __name__ == "__main__":
    demo.launch()
