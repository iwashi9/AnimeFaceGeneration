import torch
from torchvision import utils
from PIL import Image
import matplotlib.pyplot as plt

from model import StyledGenerator
from make_dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = StyledGenerator(512)
generator.load_state_dict(torch.load('weights/train_step-7.model', map_location=torch.device('cpu'))["generator"])
generator.to(device)

mean_style = None

step = 6
shape = 4 * 2 ** step

for i in range(10):
    style = generator.mean_style(torch.randn(1024, 512).to(device))
    if mean_style is None:
        mean_style = style
    else:
        mean_style += style

mean_style /= 10

def generate_image(batch_size, vec):
    vec = vec.expand(batch_size, -1).to(device)
    output = generator(
        torch.randn(batch_size, 512).to(device),
        vec,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    return output

