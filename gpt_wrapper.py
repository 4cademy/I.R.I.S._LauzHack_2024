import openai

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
    example_analyses = [
        {
            "input": {
                "metadata": {"source": "satellite_image", "location": "urban_area", "timestamp": "2023-06-15"},
                "objects": [
                    {'score': 0.9, 'label': 'building', 'box': {'xmin': 100, 'ymin': 200, 'xmax': 300, 'ymax': 400}},
                    {'score': 0.8, 'label': 'car', 'box': {'xmin': 250, 'ymin': 350, 'xmax': 350, 'ymax': 450}},
                    {'score': 0.7, 'label': 'car', 'box': {'xmin': 400, 'ymin': 300, 'xmax': 500, 'ymax': 400}}
                ]
            },
            "analysis": (
                "Analysis Summary:\n"
                "- Objects Detected: 3 objects (2 cars, 1 building)\n"
                "- Spatial Distribution: Objects are moderately clustered, with buildings dominating the background\n"
                "- Detailed Observations:\n"
                "  1. Two cars are present in the urban setting, positioned near buildings\n"
                "  2. The cars appear to be parked or stationary\n"
                "  3. Detection confidence is high, ranging from 0.7 to 0.9"
            )
        },
        {
            "input": {
                "metadata": {"source": "wildlife_camera", "location": "forest", "timestamp": "2023-09-22"},
                "objects": [
                    {'score': 0.95, 'label': 'deer', 'box': {'xmin': 200, 'ymin': 150, 'xmax': 400, 'ymax': 350}},
                    {'score': 0.6, 'label': 'tree', 'box': {'xmin': 50, 'ymin': 0, 'xmax': 600, 'ymax': 400}},
                    {'score': 0.7, 'label': 'grass', 'box': {'xmin': 0, 'ymin': 350, 'xmax': 600, 'ymax': 400}}
                ]
            },
            "analysis": (
                "Analysis Summary:\n"
                "- Objects Detected: 3 objects (1 deer, 1 tree, 1 grass area)\n"
                "- Spatial Distribution: Deer is centrally positioned with surrounding natural elements\n"
                "- Detailed Observations:\n"
                "  1. A single deer detected with high confidence (0.95)\n"
                "  2. Dense forest background with trees covering most of the image\n"
                "  3. Grass area occupies the bottom of the image\n"
                "  4. Context suggests a natural, undisturbed wildlife habitat"
            )
        }
    ]

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

# Example Usage
if __name__ == "__main__":
    # Example input data
    object_detection_data = [
        {'score': 0.5020303726196289, 'label': 'boat', 'box': {'xmin': 698, 'ymin': 263, 'xmax': 762, 'ymax': 288}},
        {'score': 0.49747616052627563, 'label': 'boat', 'box': {'xmin': 856, 'ymin': 364, 'xmax': 881, 'ymax': 408}},
        {'score': 0.4875469505786896, 'label': 'boat', 'box': {'xmin': 515, 'ymin': 276, 'xmax': 556, 'ymax': 305}},
        {'score': 0.4856327176094055, 'label': 'boat', 'box': {'xmin': 508, 'ymin': 147, 'xmax': 560, 'ymax': 182}},
        {'score': 0.4854242205619812, 'label': 'boat', 'box': {'xmin': 75, 'ymin': 500, 'xmax': 154, 'ymax': 528}}
    ]
    image_metadata = {"source": "drone_camera", "location": "harbor", "timestamp": "2024-11-30"}

    # Analyze the data
    analysis = analyze_image_data(object_detection_data, image_metadata)
    print(analysis)