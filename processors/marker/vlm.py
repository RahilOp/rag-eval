from .vlm_loader import load_qwen_vl
from PIL import Image
import os
import re
from .image_utils import resize_image, encode_image

client, model = load_qwen_vl()

def generate_response_qwen(
    image_path: str,
    prompt:str ="You will get an input image." 
                + "Image can contain either graph or text data or objects." 
                + "Extract and present all data present in the graph accurately."
                + "If image does not have graph, output the contained text or give one line description of objects."
                # + "Generate your response in Japanese language"
                # + "Generate your response in English Language"
) -> str:
    """
    This function takes an image path and an optional prompt as input and
    uses the Qwen model to generate a summary of the image in English.
    The summary is in the format of "<image_name>\n<summary_text>\n<image_name>".
    The summary_text is the output of the Qwen model, which is a text describing
    the content of the image.
    If the image does not contain a graph, the output is the text present in the image
    or a one-line description of the objects present in the image.
    """ 
    
    # # Preprocess the inputs
    # text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    # inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    # inputs = inputs.to('cuda')

    # # Inference: Generation of the output
    # output_ids = model.generate(**inputs, max_new_tokens=1024)
    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    # output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    try:
        # Encode the image
        base64_image = encode_image(image_path)
        conversation = [
                    {
                        "role":"user",
                        "content":[
                            {
                                "type":"text",
                                "text": prompt
                            },
                            {
                                "type":"image_url",
                                "image_url": {
                                    "url":  f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        ]
                    }
                ]
        
        # Send request to OpenAI
        chat_completion = client.chat.completions.create(
            messages=conversation,
            model=model,
            max_tokens=1024,
        )
        # Retrieve result from OpenAI and reset for next batch

        
        
        result = chat_completion.choices[0].message.content
    except Exception as e:
        result = f"Error: {str(e)}"
    summary = f'<{os.path.basename(image_path)}>\n'+str(result) + f'\n<{os.path.basename(image_path)}>' 

    return summary



def replace_images_with_descriptions(markdown_file_path, output_file_path) -> None:
    """
    Replaces all image links in a markdown file with their descriptions using the Qwen model.

    Args:
        markdown_file_path (str): The path to the markdown file.
        output_file_path (str): The path to the output markdown file.

    Returns:
        None

    """
    with open(markdown_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to find image links in markdown
    image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    def replace_image(match):
        image_path = match.group(1)
        image_path = image_path.lower()
        # logger.debug(f'Processing image: {image_path}')
        description = generate_response_qwen(f"{os.path.join(os.path.dirname(markdown_file_path), image_path)}")
        return description

    # Replace all image links with their descriptions
    new_content = re.sub(image_pattern, replace_image, content)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)
