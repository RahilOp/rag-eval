from PIL import Image
import base64

def resize_image(image_path):
    """Resize the image to ensure both dimensions are >= 28. And save it in the same path(basically overwrites it)"""
    image = Image.open(image_path)
    min_size = 28  # Minimum required size for both dimensions

    # Ensure the smallest dimension is at least 28, maintaining aspect ratio
    if min(image.size) < min_size:
        ratio = min_size / min(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        image.save(image_path)

# Function to encode the image
def encode_image(image_path):
    resize_image(image_path)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
