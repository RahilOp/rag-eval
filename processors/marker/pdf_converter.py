import time
import pypdfium2 # Needs to be at the top to avoid warnings
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS

import argparse
from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models

from marker.output import save_markdown

configure_logging()

MARKER_MODEL_LIST = load_all_models()

import logging
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

def process_single(
    fname= None,
    langs=['English', 'Japanese'],
    output_dir = "./processors/marker/output"
) -> None:
    """
    Process a single PDF file, convert it to markdown and save the output in
    an output directory.

    Parameters
    ----------
    fname : str
        The path to the PDF file to process.
    langs : list
        A list of languages to process the PDF in. Default is ['English', 'Japanese'].
    output_dir : str
        The directory where the output markdown files will be saved. Default is "./output".

    Returns
    -------
    None
    """
    start = time.time()
    
    full_text, images, out_meta = convert_single_pdf(
                                                    fname=fname, 
                                                    model_lst=MARKER_MODEL_LIST, 
                                                    langs=langs
                                                )

    fname = os.path.basename(fname)
    subfolder_path = save_markdown(output_dir, fname, full_text, images, out_meta)

    logger.debug(f"Saved markdown to the {subfolder_path} folder")
    logger.debug(f"Total time taken: {time.time() - start}")