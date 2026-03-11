from PIL import Image

def resize_image(image_path) -> Image:
    """Resize the image to ensure both dimensions are >= 28."""
    image = Image.open(image_path)
    min_size = 28  # Minimum required size for both dimensions

    # Ensure the smallest dimension is at least 28, maintaining aspect ratio
    if min(image.size) < min_size:
        ratio = min_size / min(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image