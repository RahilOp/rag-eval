from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def load_qwen_vl():

    # Load the model in half-precision on the available device(s)
    """
    Loads the Qwen2-VL-7B-Instruct model as an API call (already deployed on GPU)
    and returns it along with the corresponding processor.

    Returns:
        model: Qwen2VLForConditionalGeneration (loaded in GPU)
        processor: AutoProcessor
    """
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct", 
    #     torch_dtype=torch.bfloat16, 
    #     attn_implementation="flash_attention_2",
    # )
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # model.to('cuda')

    client = OpenAI(
        api_key=os.getenv("VLM_API_KEY", "your_vlm_api_key"),
        base_url=os.getenv("VLM_BASE_URL", "http://localhost:5013/v1"),
    )
    models = client.models.list()
    model = models.data[0].id

    return client, model
