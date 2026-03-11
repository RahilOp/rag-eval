import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def load_qwen_vl() -> tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:

    # Load the model in half-precision on the available device(s)
    """
    Loads the Qwen2-VL-7B-Instruct model in half-precision on the available device(s)
    and returns it along with the corresponding processor.

    Returns:
        model: Qwen2VLForConditionalGeneration (loaded in GPU)
        processor: AutoProcessor
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    model.to('cuda')

    return model, processor