import boto3
import json
import base64
from typing import List, Dict
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def process(results, image_file_path):
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
    image = Image.open(image_file_path)
    image_width, image_height = image.size

    # Calculate the width-to-height ratio
    ratio = image_width / image_height if image_height != 0 else 1  # Avoid division by zero

    # Adjust the ymin and ymax values in the bounding boxes
    for result in results:
        box = result["box"]
        box["ymin"] *= ratio
        box["ymax"] *= ratio

    return results


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

        result = process(result, image_file_path)

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

    # Iterate over results and annotate the image
    for result in results:
        if result["score"] > score_threshold:
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

    # Save the annotated image
    image.save(output_image_path)
    print(f"Annotated image saved to {output_image_path}")

    # Display the annotated image
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.axis("off")  # Hide axes
    plt.show()


image_file_path = "/Users/cloud9/Desktop/IRIS/I.R.I.S._LauzHack_2024/Screenshot 2024-11-30 at 23.32.06.jpeg"
output_image_path = "annotated_image.jpg"

results = invoke_owlv2_endpoint(image_file_path, [["arrow"]])
print(results)
annotate_and_save_image(image_file_path, results, output_image_path, score_threshold=.1)