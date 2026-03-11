from marker.convert import convert_single_pdf
from marker.models import load_all_models
import sys
import os

def extract_images_from_pdf(pdf_path):
    model_lst = load_all_models()
    full_text, images, out_meta = convert_single_pdf(pdf_path, model_lst)

    # Specify the output directory
    output_folder = './images'

    # Create the directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    image_list = []
    # Loop through the dictionary and save each image
    for filename, image in images.items():
        # Save each image to the output folder with its dictionary key as the filename
        image_path = os.path.join(output_folder, filename)
        image.save(image_path)
        image_list.append((filename, image_path))

    return image_list