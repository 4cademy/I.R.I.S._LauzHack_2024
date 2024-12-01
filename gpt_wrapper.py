import openai
from prompts import * 


def construct_prompt(data_list, metadata):
    """
    Constructs a prompt for GPT to analyze the input data with example analyses.
    Args:
        data_list (list): List of dictionaries containing object detection data.
        metadata (dict): Metadata about the image.
    
    Returns:
        str: The constructed prompt for GPT.
    """
    # Example analyses to help guide the model's response
    # Construct the prompt with examples
    prompt = "You are an AI assistant skilled at analyzing object detection data. Provide detailed, insightful analyses.\n\n"
    
    # Add example analyses
    prompt += "Example Analyses:\n\n"
    for example in example_analyses:
        prompt += f"Input Metadata: {example['input']['metadata']}\n"
        prompt += f"Input Objects: {example['input']['objects']}\n"
        prompt += f"Analysis:\n{example['analysis']}\n\n"
    
    # Add the current image's data
    prompt += "Now, analyze the following image data:\n\n"
    prompt += f"Metadata:\n{metadata}\n\n"
    prompt += f"Object Detection Data:\n{data_list}\n\n"
    
    # Prompt instructions
    prompt += (
        "Provide a comprehensive analysis with the following components:\n"
        "- A summary of the detected objects\n"
        "- Insights about the spatial distribution of objects\n"
        "- Detailed observations about the objects and their context\n"
        "- Any notable or interesting patterns\n\n"
        "Be as detailed and insightful as the example analyses."
    )

    return prompt

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
            ]
        )
        return response.choices[0].message.content
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
    print(extract_labels("Return a label"))