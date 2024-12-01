import boto3
import json
import base64
from typing import List, Dict
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from botocore.config import Config
import io 

def process(results, image):
    """
    Processes detection results by adjusting the ymin and ymax values in the bounding boxes
    based on the width-to-height ratio of the image.

    Args:
        results (list): List of detection results, where each result is a dictionary with keys:
                        - 'box' (dict): Bounding box coordinates with keys 'xmin', 'ymin', 'xmax', 'ymax'.
        image_file_path (str): Path to the image file.

    Returns:
        list: Updated detection results with adjusted ymin and ymax values.
    """
    # Open the image to get its dimensions
    image = Image.fromarray(image)
    image_width, image_height = image.size

    # Calculate the width-to-height ratio
    ratio = image_width / image_height if image_height != 0 else 1  # Avoid division by zero

    # Adjust the ymin and ymax values in the bounding boxes
    for result in results:
        box = result["box"]
        box["ymin"] *= ratio
        box["ymax"] *= ratio

    return results


def numpy_array_to_binary(np_array, format="PNG"):
    # Convert NumPy array to an image using Pillow
    image = Image.fromarray(np_array)

    # Save the image to a BytesIO buffer in the specified format
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    # Read binary data from the buffer
    return buffer.read()

def numpy_array_to_base64(np_array, format="PNG"):
    image_pil = Image.fromarray(np_array)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")  # You can change the format if needed
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64

def invoke_owlv2_endpoint(image: np.array, labels: List[str], endpoint_name="huggingface-pytorch-inference-2024-11-30-19-33-00-339") -> Dict:
    """
    Calls a SageMaker endpoint with an image file and a list of labels for inference.

    Args:
        file_path (str): The file path to the image.
        labels (List[str]): A list of labels (strings) for classification or object detection.
        endpoint_name (str): The name of the SageMaker endpoint.

    Returns:
        Dict: The inference results returned by the model.
    """
    
    runtime = boto3.client('sagemaker-runtime')

    try:
        # Read the image in binary format
        # with open(file_path, "rb") as f:
        #     image_binary = f.read()

        image_binary = numpy_array_to_binary(image)

        # Prepare the payload
        payload = {
            
            "image": base64.b64encode(image_binary).decode("utf-8"),  # Encode image as base64
            "candidate_labels": labels  # Pass the list of labels
        }

        # Convert the payload to a JSON string
        payload_json = json.dumps(payload)

        # Invoke the SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload_json,
        )
        
        # Parse and return the response
        result = json.loads(response["Body"].read().decode('utf-8'))

        result = process(result, image)

        return result

    except Exception as e:
        print(f"Error invoking SageMaker endpoint: {e}")
        return {"error": str(e)}


def annotate_image(image, results, score_threshold=0.05):
    """
    Annotates an image with bounding boxes based on detection results and saves it to an output file.

    Args:
        image_path (str): Path to the input image.
        results (list): List of detection results, where each result is a dictionary with keys:
                        - 'score' (float): Confidence score of the detection.
                        - 'label' (str): Label of the detected object.
                        - 'box' (dict): Bounding box coordinates with keys 'xmin', 'ymin', 'xmax', 'ymax'.
        output_image_path (str): Path to save the annotated image.
        score_threshold (float): Minimum confidence score required to draw a bounding box.
    """
    # Load the image
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    # Iterate over results and annotate the image
    for result in results:
            print(type(result))
        # if float(result["score"]) > score_threshold:
            box = result['box']
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

            # Draw the bounding box
            draw.rectangle(
                (xmin, ymin, xmax, ymax), 
                outline="red", 
                width=3
            )
            # Optionally, you can add labels or confidence scores (uncomment the line below if needed)
            # draw.text((xmin, ymin * ratio), f"{result['label']}: {round(result['score'], 2)}", fill="white")

    return image

def save_image(image, output_image_path):
    # Save the annotated image
    image.save(output_image_path)
    print(f"Annotated image saved to {output_image_path}")

    # Display the annotated image
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.axis("off")  # Hide axes
    plt.show()


import ast

image_file_path = "/Users/cloud9/Desktop/IRIS/I.R.I.S._LauzHack_2024/examples/traunkirchen-road-tunnel-with-cars-austria-obersterreich-upper-austria-GGPCR1.jpg"
output_image_path = "annotated_image.jpg"
image = Image.open(image_file_path)
image = np.array(image)
results = invoke_owlv2_endpoint(image, [["car"]])

print(results)

# save_image(annotate_image(image, results, output_image_path), output_image_path)