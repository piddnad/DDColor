import cv2
import numpy as np
import torch
import torch.nn.functional as F


def load_checkpoint_state_dict(model_path: str, map_location="cpu"):
    """Load a checkpoint and return a state_dict.

    Supports both:
    - {'params': state_dict, ...} (common in this repo)
    - raw state_dict
    """
    ckpt = torch.load(model_path, map_location=map_location)
    if isinstance(ckpt, dict) and "params" in ckpt:
        return ckpt["params"]
    return ckpt


def build_ddcolor_model(
    model_cls,
    *,
    model_path: str,
    input_size: int = 512,
    model_size: str = "large",
    decoder_type: str = "MultiScaleColorDecoder",
    device=None,
    **kwargs,
):
    """Build a DDColor model and load weights.

    This helper is intentionally backend-agnostic: `model_cls` can be
    `ddcolor.DDColor` or `basicsr.archs.ddcolor_arch.DDColor` as long as
    it supports the common constructor args used below.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_size not in ("tiny", "large"):
        raise ValueError(f"model_size must be 'tiny' or 'large', got: {model_size}")
    encoder_name = "convnext-t" if model_size == "tiny" else "convnext-l"

    if decoder_type == "MultiScaleColorDecoder":
        # keep default consistent with existing scripts
        kwargs.setdefault("num_queries", 100)
        kwargs.setdefault("num_scales", 3)
        kwargs.setdefault("dec_layers", 9)
    elif decoder_type == "SingleColorDecoder":
        kwargs.setdefault("num_queries", 256)
    else:
        raise NotImplementedError(f"decoder_type not implemented: {decoder_type}")

    model = model_cls(
        encoder_name=encoder_name,
        decoder_name=decoder_type,
        input_size=[input_size, input_size],
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        **kwargs,
    )

    state_dict = load_checkpoint_state_dict(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


class ColorizationPipeline:
    """Shared image colorization pipeline used by CLI/Gradio/Cog.

    - input: BGR uint8 image (OpenCV)
    - output: BGR uint8 image (OpenCV)
    """

    def __init__(self, model, *, input_size: int = 512, device=None):
        self.input_size = int(input_size)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            if img_bgr is None:
                raise ValueError("img is None (cv2.imread failed?)")

            height, width = img_bgr.shape[:2]

            img = (img_bgr / 255.0).astype(np.float32)
            orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

            # resize rgb image -> lab -> get grey -> rgb
            img_resized = cv2.resize(img, (self.input_size, self.input_size))
            img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
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

            output_ab = self.model(tensor_gray_rgb).cpu()  # (1, 2, input_size, input_size)

            # resize ab -> concat original l -> bgr
            output_ab_resized = (
                F.interpolate(output_ab, size=(height, width))[0]
                .float()
                .numpy()
                .transpose(1, 2, 0)
            )
            output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
            output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

            output_img = (output_bgr * 255.0).round().astype(np.uint8)
            return output_img
