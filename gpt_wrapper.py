import openai
from prompts import * 
from vision import numpy_array_to_base64

from PIL import Image
import numpy as np

# The rest of the code remains the same as in the previous version
def call_openai_api(prompt, model="gpt-4o"):
    """
    Calls the OpenAI API with the given prompt.
    Args:
        prompt (str): The prompt to send to the OpenAI API.
        model (str): The model to use (default: gpt-4o).
    
    Returns:
        str: The response from the OpenAI API.
    """
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant analyzing image data."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        # return response.choices[0].message.content
        return response
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

def analyze_image_data(data_list, metadata):
    """
    Analyzes image data using the OpenAI API.
    Args:
        data_list (list): List of dictionaries containing object detection data.
        metadata (dict): Metadata about the image.
    
    Returns:
        str: The analysis from GPT.
    """
    prompt = construct_prompt(data_list, metadata)
    return call_openai_api(prompt)


def extract_labels(prompt, model="gpt-3.5-turbo"):
    # Extract object labels from the sentence: "How many ships are there on the image?"
    # Output: ["ship"]

    """
    Calls the OpenAI API with the given prompt.
    Args:
        prompt (str): The prompt to send to the OpenAI API.
        model (str): The model to use (default: gpt-4o).
    
    Returns:
        str: The response from the OpenAI API.
    """

    extract_labels_messages.append({"role": "user", "content": prompt})

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages = extract_labels_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def analyze_image(prompt, image, model="gpt-4o"):

    base64_image = numpy_array_to_base64(image)

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url","image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]
        )
        # response = openai.Image.create(
        #     image=image_data,
        #     prompt="Describe the contents of this image.",
        #     n=1,
        #     size="1024x1024"
        # )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


# Example Usage
if __name__ == "__main__":
    # Example input data
    # object_detection_data = [
    #     {'score': 0.5020303726196289, 'label': 'boat', 'box': {'xmin': 698, 'ymin': 263, 'xmax': 762, 'ymax': 288}},
    #     {'score': 0.49747616052627563, 'label': 'boat', 'box': {'xmin': 856, 'ymin': 364, 'xmax': 881, 'ymax': 408}},
    #     {'score': 0.4875469505786896, 'label': 'boat', 'box': {'xmin': 515, 'ymin': 276, 'xmax': 556, 'ymax': 305}},
    #     {'score': 0.4856327176094055, 'label': 'boat', 'box': {'xmin': 508, 'ymin': 147, 'xmax': 560, 'ymax': 182}},
    #     {'score': 0.4854242205619812, 'label': 'boat', 'box': {'xmin': 75, 'ymin': 500, 'xmax': 154, 'ymax': 528}}
    # ]
    # image_metadata = {"source": "drone_camera", "location": "harbor", "timestamp": "2024-11-30"}

    # # Analyze the data
    # analysis = analyze_image_data(object_detection_data, image_metadata)
    # print(analysis)
    image_path = "/Users/cloud9/Desktop/IRIS/I.R.I.S._LauzHack_2024/Screenshot 2024-11-30 at 22.37.15.png"  # Replace with the path to your image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    print(analyze_image("Describe this image", image_array))
    # print(extract_labels("Return a label"))