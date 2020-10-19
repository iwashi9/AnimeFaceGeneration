import torch
from torchvision import utils
from PIL import Image
import matplotlib.pyplot as plt

from model import StyledGenerator
from make_dataset import *


generator = StyledGenerator(512)
generator.load_state_dict(torch.load('weights/train_step-7.model', map_location=torch.device('cpu'))["generator"])

mean_style = None

step = 6
shape = 4 * 2 ** step

for i in range(10):
    style = generator.mean_style(torch.randn(1024, 512))
    if mean_style is None:
        mean_style = style
    else:
        mean_style += style

mean_style /= 10

def generate_image(batch_size, vec):
    vec = vec.expand(batch_size, -1)
    output = generator(
        torch.randn(batch_size, 512),
        vec,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    return output

