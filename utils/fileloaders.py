import os
from PIL import Image
import numpy as np

def load_image(file_name):
    image = Image.open(file_name)
    return image


