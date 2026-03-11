import os
from .vlm import replace_images_with_descriptions
from .pdf_converter import process_single

langs = ['English', 'Japanese']
output_dir = 'output'

def pipeline(file_path, output_dir:str = './processors/marker/output') -> None:
    
    """
    Process a PDF file and convert it to markdown, replace images with descriptions using VLM model and save the output markdown.

    Parameters
    ----------
    file_path : str
        The path to the PDF file to process.
    output_dir : str
        The directory where the output markdown files will be saved. Default is "./output".

    Returns
    -------
    None

    """
    file_name = os.path.basename(file_path)
    output_dir = output_dir

    markdown_file_path = f"{output_dir}/{file_name.split('.')[0]}/{file_name.split('.')[0]}.md"
    output_file_path = f"{output_dir}/{file_name.split('.')[0]}/{file_name.split('.')[0]}_out.md"
    
    process_single(
        fname=file_path, 
        langs=['English', 'Japanese'],
        output_dir=output_dir
    )

    replace_images_with_descriptions(
        markdown_file_path=markdown_file_path,
        output_file_path=output_file_path        
    )
    
    print("Output markdown has been created successfully")

if __name__ == "__main__":
    pipeline(file_path="./pdf/pdfsmall.pdf", output_dir="./output_test")
    