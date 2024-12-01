import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np

class EmbeddingExtractor:
    def __init__(self, model_name="mobilenet_v2"):
        # Load a pre-trained model (e.g., MobileNetV2)
        if model_name == "mobilenet_v2":
            base_model = models.mobilenet_v2(pretrained=True)
            self.model = nn.Sequential(*list(base_model.children())[:-1])  # Remove the classifier
        elif model_name == "efficientnet_b0":
            base_model = models.efficientnet_b0(pretrained=True)
            self.model = nn.Sequential(*list(base_model.children())[:-1])  # Remove the classifier
        else:
            raise ValueError("Unsupported model name")

        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image):
        """
        Extracts feature embedding from an image.
        :param image: PIL Image
        :return: Flattened feature embedding as a torch tensor
        """
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = self.model(image_tensor).squeeze()  # Remove batch and additional dimensions
        return embedding.flatten()  # Ensure embedding is a 1D tensor

def cosine_similarity(embedding1, embedding2):
    """
    Computes cosine similarity between two embeddings.
    :param embedding1: Torch tensor (1D)
    :param embedding2: Torch tensor (1D)
    :return: Cosine similarity score
    """
    return torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

def filter_patches_with_embeddings(detection, image_path, reference_image_path, model_name="mobilenet_v2", threshold=0.3):
    """
    Filters detected patches by comparing their embeddings to a reference image.
    :param detection: Dictionary with 'scores', 'labels', and 'boxes' tensors.
    :param image_path: Path to the original image.
    :param reference_image_path: Path to the reference image.
    :param model_name: Name of the model for embedding extraction.
    :param threshold: Minimum similarity score to keep a detection.
    :return: Filtered detections.
    """
    # Load the images
    image = Image.fromarray(image_path).convert("RGB")
    reference_image = Image.fromarray(reference_image_path).convert("RGB")
    extractor = EmbeddingExtractor(model_name=model_name)

    # Get reference image embedding
    reference_embedding = extractor.get_embedding(reference_image)

    print(f"detection passed: {detection}")

    filtered_detections = []
    draw = ImageDraw.Draw(image)

    for result in detection:

        box = result['box']
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        cropped_region = image.crop((xmin, ymin, xmax, ymax))

        # Get embedding for the cropped region
        try:
            cropped_embedding = extractor.get_embedding(cropped_region)
            similarity = cosine_similarity(cropped_embedding, reference_embedding)

            print(similarity)
            # Keep detection if similarity exceeds the threshold
            if similarity > threshold:
                filtered_detections.append({'score': result['score'], 'label':result['label'], 'box':result['box']})

                # Draw the box and similarity score on the image
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="green", width=2)
                draw.text((xmin, ymin - 10), f"Sim: {similarity:.2f}", fill="green")

        except Exception as e:
            print(f"Error processing patch: {e}")

    # Show or save the filtered image
    # image.show()
    print(f"filtered detections {filtered_detections}")
    return filtered_detections


# Example usage
# filtered_results = filter_patches_with_embeddings(detection, image_path, reference_image_path, model_name="mobilenet_v2", threshold=0.5)
