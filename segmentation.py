import io

import numpy as np
import boto3
import json
import base64
from vision import invoke_owlv2_endpoint, annotate_image, numpy_array_to_base64, numpy_array_to_binary
# import time
import time
import torchvision
from PIL import Image, ImageDraw

from vision import invoke_owlv2_endpoint

# SageMaker endpoints for CV and segmentation models
cv_endpoint = "huggingface-pytorch-inference-2024-11-30-19-33-00-339"  # Replace with your CV model endpoint name
segmentation_endpoint = "huggingface-pytorch-inference-2024-11-30-20-23-35-096"  # Replace with your segmentation model endpoint name


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

# Function to invoke segmentation model
def invoke_segmentation_model(image, endpoint_name, boxes):
    """
    Invokes the segmentation model on a given cropped image.

    Args:
        image_array (np.array): The image to process as a NumPy array.
        endpoint_name (str): The SageMaker endpoint name for the segmentation model.
        boxes (list): Bounding boxes to process.

    Returns:
        dict: Segmentation results including the segmentation mask.
    """
    runtime = boto3.client('sagemaker-runtime')

    try:
        #  # Ensure image is a PIL Image
        # if isinstance(image, np.ndarray):
        #     image = Image.fromarray(image)

        # # Convert the PIL image to bytes
        # image_buffer = io.BytesIO()
        # image.save(image_buffer, format="JPEG")
        # image_binary = image_buffer.getvalue()

        # Open the image file and convert it to binary
        with open(image, "rb") as image_file:
            image_binary = image_file.read()

        # Encode the image as Base64
        encoded_image = base64.b64encode(image_binary).decode("utf-8")

        # Prepare the payload
        payload = {"image": encoded_image}
        if boxes:
            payload["input_boxes"] = boxes

        # Convert the payload to a JSON string
        payload_json = json.dumps(payload)

        # Invoke the SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload_json,
        )

        # Parse and return the response
        result = json.loads(response["Body"].read().decode("utf-8"))
        return result

    except Exception as e:
        print(f"Error invoking SageMaker endpoint: {e}")
        return {"error": str(e)}




# Updated function to process image
def process_image_with_segmentation(image_array):
    """
    Processes an image through the CV model and segmentation model.

    Args:
        image_array (np.array): The input image as a NumPy array.

    Returns:
        list: Combined results with bounding boxes and segmentation masks.
    """
    # Step 1: Invoke the CV model
    results = invoke_owlv2_endpoint(image_array, [["car"]])
    print("CV Model Results:", json.dumps(results, indent=2))

    # Handle errors in CV model invocation
    if "error" in results:
        print(f"Error invoking CV model: {results['error']}")
        return []

    # Process the results into bounding boxes
    boxes = []
    try:
        for result in results:
            box = result['box']  # Access the 'box' dictionary
            xyxy = [float(box['xmin']), float(box['ymin']), float(box['xmax']), float(box['ymax'])]
            boxes.append(xyxy)  # Append to the list
    except (KeyError, TypeError) as e:
        print(f"Error processing bounding boxes: {e}")
        return []

    print("Processed bounding boxes:", boxes)
    return boxes


def draw_boxes_and_segmentation(image_file_path, output_image_path, segmentation_results):
    """
    Draws bounding boxes and segmentation masks on the image and saves it to an output file.

    Args:
        image_file_path (str): Path to the input image.
        output_image_path (str): Path to save the annotated image.
        segmentation_results (list): Segmentation results including bounding boxes and masks.
    """
    # Load the image
    image = Image.open(image_file_path).convert("RGBA")  # Convert to RGBA for overlay
    draw = ImageDraw.Draw(image)

    # Iterate over segmentation results
    for result in segmentation_results:
        # Draw bounding boxes
        box = result.get("box", {})
        if box:
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        # Draw segmentation masks if present
        if "mask" in result:
            mask_array = np.array(result["mask"])  # Assuming the mask is in NumPy array format
            mask = Image.fromarray((mask_array * 255).astype("uint8"))  # Convert mask to binary image
            mask = mask.convert("RGBA")  # Convert mask to RGBA format
            image.paste(mask, (0, 0), mask)  # Overlay the mask

    # Save the annotated image
    image.save(output_image_path)
    print(f"Annotated image saved at: {output_image_path}")




# Test the pipeline
image_file_path = "C:\\Users\\krish\\Documents\\Hackathons\\LauzHack2024\\parking_lot.jpg"
image = Image.open(image_file_path)
image_array = np.array(image)
output_image_path = "output_with_segmentation.jpg"

# Process the image through the pipeline
boxes = process_image_with_segmentation(image_array)
if boxes:
    segmentation_results = invoke_segmentation_model(image_file_path, segmentation_endpoint, boxes)
    if segmentation_results and "error" not in segmentation_results:
        draw_boxes_and_segmentation(image_array, output_image_path, segmentation_results)
    else:
        print("Error with segmentation results:", segmentation_results)
else:
    print("Error processing image for segmentation.")
