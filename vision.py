import boto3
import json
import base64
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont
import random


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
        result = json.loads(response["Body"].read().decode("utf-8"))
        return result

    except Exception as e:
        print(f"Error invoking SageMaker endpoint: {e}")
        return {"error": str(e)}


def draw_boxes_on_image(image_path: str, output_path: str, detections_json: str):
    """
    Draws bounding boxes with labels and scores on the given image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
        detections_json (str): JSON string of detection results with scores, labels, and bounding boxes.

    Returns:
        None
    """
    # with torch.no_grad():
    # outputs = model(**inputs)
    # target_sizes = torch.tensor([im.size[::-1]])
    # results = processor.post_process_object_detection(outputs, threshold=0.05, target_sizes=target_sizes)[0]

    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)

    scores = results["scores"].tolist()
    labels = results["labels"].tolist()
    boxes = results["boxes"].tolist()

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")

    # Parse the JSON string into a Python object
    detections = json.loads(detections_json)

    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Define font for text (fallback if the font cannot be loaded)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Assign random colors for each label
    label_colors = {}

    for detection in detections:
        label = detection["label"]
        score = detection["score"]
        box = detection["box"]

        # Generate a random color if the label doesn't have one
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(0, 255) for _ in range(3))
        
        color = label_colors[label]
        
        # Draw the bounding box
        draw.rectangle(
            [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
            outline=color,
            width=3
        )

        # Add label and score text
        text = f"{label}: {score:.2f}"
        # Calculate text size using textbbox
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Draw background for text
        text_background = [
            (box["xmin"], box["ymin"] - text_height),
            (box["xmin"] + text_width, box["ymin"])
        ]
        draw.rectangle(text_background, fill=color)  # Text background
        draw.text((box["xmin"], box["ymin"] - text_height), text, fill="white", font=font)
    
    # Save the image
    image.save(output_path)
    print(f"Annotated image saved to {output_path}")


image_file_path = "/Users/cloud9/Desktop/IRIS/I.R.I.S._LauzHack_2024/vessels/2024-08-22-00_00_2024-08-22-23_59_Sentinel-2_L2A_True_color (2).jpg"
results = invoke_owlv2_endpoint(image_file_path, ["ship"])
detections = json.dumps(results, indent=2)

print(detections)

output_image_path = "annotated_image.jpg"
draw_boxes_on_image(image_file_path, output_image_path, detections)
