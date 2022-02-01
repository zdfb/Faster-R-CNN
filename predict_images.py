from PIL import Image
from utils.utils_frcnn import FRCNN

image_path = 'Image_samples/street.jpg'  # 测试图片路径

frcnn = FRCNN()

image = Image.open(image_path)
image = frcnn.detect_image(image)
image.show()