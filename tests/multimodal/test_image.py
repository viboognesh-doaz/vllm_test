from PIL import Image, ImageChops
from pathlib import Path
from vllm.multimodal.image import convert_image_mode
import numpy as np
ASSETS_DIR = Path(__file__).parent / 'assets'
assert ASSETS_DIR.exists()

def test_rgb_to_rgb():
    original_image = Image.open(ASSETS_DIR / 'image1.png').convert('RGB')
    converted_image = convert_image_mode(original_image, 'RGB')
    diff = ImageChops.difference(original_image, converted_image)
    assert diff.getbbox() is None

def test_rgba_to_rgb():
    original_image = Image.open(ASSETS_DIR / 'rgba.png')
    original_image_numpy = np.array(original_image)
    converted_image = convert_image_mode(original_image, 'RGB')
    converted_image_numpy = np.array(converted_image)
    for i in range(original_image_numpy.shape[0]):
        for j in range(original_image_numpy.shape[1]):
            if original_image_numpy[i][j][3] == 0:
                assert converted_image_numpy[i][j][0] == 255
                assert converted_image_numpy[i][j][1] == 255
                assert converted_image_numpy[i][j][2] == 255