import boto3
import json
import base64
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont
import random

import matplotlib.pyplot as plt


def invoke_owlv2_endpoint(file_path: str, labels: List[str], endpoint_name="huggingface-pytorch-inference-2024-11-30-19-33-00-339") -> Dict:
    """
    Calls a SageMaker endpoint with an image file and a list of labels for inference.

    Args:
        file_path (str): The file path to the image.
        labels (List[str]): A list of labels (strings) for classification or object detection.
        endpoint_name (str): The name of the SageMaker endpoint.

    Returns:
        Dict: The inference results returned by the model.
    """
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime')

    try:
        # Read the image in binary format
        with open(file_path, "rb") as f:
            image_binary = f.read()

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
        return result

    except Exception as e:
        print(f"Error invoking SageMaker endpoint: {e}")
        return {"error": str(e)}


def annotate_and_save_image(image_path, results, output_image_path, score_threshold=0.05):
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
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Get image dimensions
    image_width, image_height = image.size

    # Iterate over results and annotate the image
    for result in results:
        if result["score"] > score_threshold:
            box = result['box']
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            
            # Calculate the width-to-height ratio
            ratio = image_width / image_height if image_height != 0 else 1  # Avoid division by zero

            # Draw the bounding box
            draw.rectangle(
                (xmin, ymin * ratio, xmax, ymax * ratio), 
                outline="red", 
                width=1
            )
            # Optionally, you can add labels or confidence scores (uncomment the line below if needed)
            # draw.text((xmin, ymin * ratio), f"{result['label']}: {round(result['score'], 2)}", fill="white")

    # Save the annotated image
    image.save(output_image_path)
    print(f"Annotated image saved to {output_image_path}")

    # Display the annotated image
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.axis("off")  # Hide axes
    plt.show()


image_file_path = "/Users/cloud9/Desktop/IRIS/I.R.I.S._LauzHack_2024/vessels/2024-08-22-00_00_2024-08-22-23_59_Sentinel-2_L2A_True_color.jpg"
results = invoke_owlv2_endpoint(image_file_path, [["boat"]])
output_image_path = "annotated_image.jpg"
print(results)
annotate_and_save_image(image_file_path, results, output_image_path)

# image = Image.open(image_file_path)

# draw = ImageDraw.Draw(image)

# for result in results:
#     if result["score"] > .05:

#         box = result['box']
#         xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        
#         # Calculate the width-to-height ratio
#         width, height = image.size
#         ratio = width / height if height != 0 else 1  # Avoid division by zero

#         # Use the ratio for scaling
#         draw.rectangle(
#             (xmin, ymin * ratio, xmax, ymax * ratio), 
#             outline="red", 
#             width=1
#         )
# fig, ax = plt.subplots(1)
# ax.imshow(image)
# plt.show()