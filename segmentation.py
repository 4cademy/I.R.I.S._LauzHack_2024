import io
import boto3
import json
import base64
import time
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
        image (PIL.Image): The cropped image to process.
        endpoint_name (str): The SageMaker endpoint name for the segmentation model.

    Returns:
        dict: Segmentation results including the segmentation mask.
    """
    runtime = boto3.client('sagemaker-runtime')
    

    try:
            # Read the image in binary format
            with open(image, "rb") as f:
                image_binary = f.read()

            # Prepare the payload
            payload = {
                "image": base64.b64encode(image_binary).decode("utf-8"),  # Encode image as base64
                "input_boxes": boxes  # Pass the bounding boxes
                
            }

            # Convert the payload to a JSON string
            payload_json = json.dumps(payload)

            start_time = time.time()
            # Invoke the SageMaker endpoint
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=payload_json,
            )
            elapsed_time = time.time() - start_time
            print(f"Segmentation model took {elapsed_time:.2f} seconds.")
            
            # Parse and return the response
            result = json.loads(response["Body"].read().decode('utf-8'))

            result = process(result, image_file_path)

            return result

    except Exception as e:
            print(f"Error invoking SageMaker endpoint: {e}")
            return {"error": str(e)}


        
# Updated function to process image
def process_image_with_segmentation(image_file_path):
    """
    Processes an image through the CV model and segmentation model.

    Args:
        image_file_path (str): Path to the input image.

    Returns:
        list: Combined results with bounding boxes and segmentation masks.
    """
    # Step 1: Invoke the CV model
    results = invoke_owlv2_endpoint(image_file_path, ["ship"], cv_endpoint)
    print("CV Model Results:", json.dumps(results, indent=2))
    

    boxes = []
    for result in results:
        box = result['box']  # Access the 'box' dictionary
        xyxy = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]  # Convert to [x_min, y_min, x_max, y_max]
        boxes.append(xyxy)  # Append to the list
    return boxes

def draw_boxes_and_segmentation(image_file_path, output_image_path, segmentation_results):
    """
    Draws bounding boxes and segmentation masks on the image and saves it to an output file.

    Args:
        image_file_path (str): Path to the input image.
        output_image_path (str): Path to save the annotated image.
        segmentation_results (dict): Segmentation results including the segmentation mask.
    """
    # Load the image
    image = Image.open(image_file_path)
    draw = ImageDraw.Draw(image)
    
    # Draw bounding boxes
    for result in segmentation_results:
        box = result['box']
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

    # Draw segmentation masks
    mask = Image.fromarray(segmentation_results["mask"])
    image.paste(mask, (0, 0), mask)

    # Save the annotated image
    image.save(output_image_path)
    print(f"Annotated image saved at: {output_image_path}")



# Test the pipeline
image_file_path = "C:\\Users\\krish\\Documents\\Hackathons\\LauzHack2024\\parking_lot.jpg"
output_image_path = "output_with_segmentation.jpg"

# Process the image through the pipeline
boxes = process_image_with_segmentation(image_file_path)
segmentation_results = invoke_segmentation_model(image_file_path, segmentation_endpoint, boxes)

# Draw and save the annotated image
draw_boxes_and_segmentation(image_file_path, output_image_path, segmentation_results)