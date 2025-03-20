from PIL import Image
import numpy as np
from rembg import remove

class PreprocessInput:
    def __init__(self):
        self.image = None

    def remove_background(self, file_path: str):
        img = Image.open(file_path)
        img_no_bg = remove(img)
        return np.asarray(img_no_bg)

    def resize(self, img, width=768, height=1024):
        return img.resize((width, height))