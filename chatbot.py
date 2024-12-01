import openai
import boto3
import json
import os
from prompts import *
from vision import numpy_array_to_base64

from PIL import Image
import numpy as np

# Initialize the Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
)

def extract_labels(prompt, model="gpt-3.5-turbo"):
    """
    Extract object labels from the prompt using the OpenAI API.
    Args:
        prompt (str): The prompt to send to the OpenAI API.
        model (str): The model to use (default: gpt-3.5-turbo).

    Returns:
        str: The extracted labels from the API response.
    """
    try:
        extract_labels_messages = [
            {"role": "system", "content": "You are an AI assistant extracting labels from text."},
            {"role": "user", "content": prompt}
        ]
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=extract_labels_messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error extracting labels: {e}"


def define_kwargs(model_id, prompt=None, messages=None):
    """
    Constructs the request parameters for invoking the Bedrock model.

    Args:
        model_id (str): The identifier of the model to be used.
        prompt (str): The input prompt for the model.
        messages (list): Existing messages for a conversation (default: None).

    Returns:
        dict: The request parameters to invoke the Bedrock model.
    """
    input_messages = messages if messages else []
    input_messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    return {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": 1,
            "top_p": 0.999,
            "messages": input_messages
        })
    }


def stream_bedrock_response(prompt=None, messages=None, image=None, model_id="anthropic.claude-3-5-sonnet-20241022-v2:0"):
    """
    Sends a prompt to a Bedrock model and streams the response.

    Args:
        prompt (str): The input prompt to send to the model.
        messages (list): Existing messages for a conversation (default: None).
        model_id (str): The model ID to use for the request (default: Claude 3.5).

    Returns:
        object: The response stream object from the Bedrock model.
    """
    kwargs = define_kwargs(model_id, prompt=prompt, messages=messages)
    try:
        response = bedrock_runtime.invoke_model_with_response_stream(**kwargs)
        return response.get("body")
    except Exception as e:
        print(f"An error occurred while streaming the response: {e}")


def describe_image_openai(image, model="gpt-4o"):
    """
    Analyzes an image by sending it to the OpenAI API in Base64 format.
    Args:
        prompt (str): The prompt for the image analysis.
        image (np.ndarray): The image as a NumPy array.
        model (str): The model to use (default: gpt-4o).

    Returns:
        str: The analysis result from the API.
    """
    base64_image = numpy_array_to_base64(image)
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_describe_image},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {e}"


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


if __name__ == "__main__":
    # Example OpenAI image analysis
    image_path = "/Users/cloud9/Desktop/IRIS/I.R.I.S._LauzHack_2024/Screenshot 2024-11-30 at 22.37.15.png"
    
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        result = describe_image_openai(image_array)
        print(result)
    except FileNotFoundError:
        print("Error: The specified image file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    # Example Bedrock response
    # stream = stream_bedrock_response("How many cars are in the image")
    # if stream:
    #     for chunk in stream:
    #         print(chunk.decode('utf-8'))
