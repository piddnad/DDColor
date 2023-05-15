from torchvision.transforms import ToTensor, Grayscale


def color_enhacne_blend(x, factor=1.2):
    x_g = Grayscale(3)(x)
    out = x_g * (1.0 - factor) + x * factor
    out[out < 0] = 0
    out[out > 1] = 1
    return out