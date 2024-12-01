extract_labels_messages = [
                            {"role": "system", "content": 'Extract the object labels explicitly mentioned in the sentences below. Focus only on the specific objects or entities the user is asking about (e.g., "ship," "car," "tree"). Ignore locations, places, or additional context. Return the labels in a list format.'},
                            {"role": "user", "content": "How many ships are there on the image?"},
                            {"role": "assistant", "content": '["ship"]'},
                            {"role": "user", "content": "What is the color of the cars visible in the picture?"},
                            {"role": "assistant", "content": '["car"]'},
                            {"role": "user", "content": "Count all the airplanes flying in the sky."},
                            {"role": "assistant", "content": '["airplane"]'},
                            {"role": "user", "content": "Are there any birds perched on the tree branches?"},
                            {"role": "assistant", "content": '["bird"]'},
                            {"role": "user", "content": "How many chairs and tables are arranged in the room?"},
                            {"role": "assistant", "content": '["chair", "table"]'},
                            {"role": "user", "content": "What animals can you see in this photo?"},
                            {"role": "assistant", "content": '["animal"]'},
                            {"role": "user", "content": "Do you see any books or pens on the desk?"},
                            {"role": "assistant", "content": '["book", "pen"]'},
                            {"role": "user", "content": "How many boats are anchored at the dock?"},
                            {"role": "assistant", "content": '["boat"]'},
                            {"role": "user", "content": "Are there any flowers blooming in the garden?"},
                            {"role": "assistant", "content": '["flower"]'},
                            {"role": "user", "content": "Can you count the bicycles parked outside?"},
                            {"role": "assistant", "content": '["bicycle"]'},
                            {"role": "user", "content": "What type of vehicles are shown in the garage?"},
                            {"role": "assistant", "content": '["vehicle"]'},
                            {"role": "user", "content": "Is there a computer or monitor on the desk?"},
                            {"role": "assistant", "content": '["computer", "monitor"]'},
                            {"role": "user", "content": "Are there any trees or mountains in the background?"},
                            {"role": "assistant", "content": '["tree", "mountain"]'},
                            {"role": "user", "content": "How many people are standing near the train?"},
                            {"role": "assistant", "content": '["person"]'},
                            {"role": "user", "content": "Can you identify the fruits kept in the basket?"},
                            {"role": "assistant", "content": '["fruit"]'},
                            {"role": "user", "content": "What kind of animals are walking on the street?"},
                            {"role": "assistant", "content": '["animal"]'},
                            {"role": "user", "content": "How many clouds are in the sky above the houses?"},
                            {"role": "assistant", "content": '["cloud"]'},
                            {"role": "user", "content": "Do you notice any stars or the moon in the night sky?"},
                            {"role": "assistant", "content": '["star", "moon"]'},
                            {"role": "user", "content": "Is there a river flowing beside the mountains?"},
                            {"role": "assistant", "content": '["river"]'},
                            {"role": "user", "content": "Can you see a cat or dog in this photo?"},
                            {"role": "assistant", "content": '["cat", "dog"]'},
                            {"role": "user", "content": "Are there any bookshelves in the background?"},
                            {"role": "assistant", "content": '["bookshelf"]'},
                            {"role": "user", "content": "What objects are visible in this image?"},
                            {"role": "assistant", "content": '["object"]'},
                            {"role": "user", "content": "Are there any bottles and glasses on the table?"},
                            {"role": "assistant", "content": '["bottle", "glass"]'}, 
                        ]


extract_labels_messages_claude = [
                            # {"role": "required", "content": "Extract the object labels explicitly mentioned in the sentences below. Focus only on the specific objects or entities the user is asking about (e.g., 'ship,' 'car,' 'tree'). Ignore locations, places, or additional context. Return the labels in a list format."},
                            {"role": "user", "content": [{"type": "text", "text": "How many ships are there on the image?"}]},
                            {"role": "assistant", "content": '["ship"]'},
                            {"role": "user", "content": [{"type": "text", "text": "What is the color of the cars visible in the picture?"}]},
                            {"role": "assistant", "content": '["car"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Count all the airplanes flying in the sky."}]},
                            {"role": "assistant", "content": '["airplane"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Are there any birds perched on the tree branches?"}]},
                            {"role": "assistant", "content": '["bird"]'},
                            {"role": "user", "content": [{"type": "text", "text": "How many chairs and tables are arranged in the room?"}]},
                            {"role": "assistant", "content": '["chair", "table"]'},
                            {"role": "user", "content": [{"type": "text", "text": "What animals can you see in this photo?"}]},
                            {"role": "assistant", "content": '["animal"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Do you see any books or pens on the desk?"}]},
                            {"role": "assistant", "content": '["book", "pen"]'},
                            {"role": "user", "content": [{"type": "text", "text": "How many boats are anchored at the dock?"}]},
                            {"role": "assistant", "content": '["boat"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Are there any flowers blooming in the garden?"}]},
                            {"role": "assistant", "content": '["flower"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Can you count the bicycles parked outside?"}]},
                            {"role": "assistant", "content": '["bicycle"]'},
                            {"role": "user", "content": [{"type": "text", "text": "What type of vehicles are shown in the garage?"}]},
                            {"role": "assistant", "content": '["vehicle"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Is there a computer or monitor on the desk?"}]},
                            {"role": "assistant", "content": '["computer", "monitor"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Are there any trees or mountains in the background?"}]},
                            {"role": "assistant", "content": '["tree", "mountain"]'},
                            {"role": "user", "content": [{"type": "text", "text": "How many people are standing near the train?"}]},
                            {"role": "assistant", "content": '["person"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Can you identify the fruits kept in the basket?"}]},
                            {"role": "assistant", "content": '["fruit"]'},
                            {"role": "user", "content": [{"type": "text", "text": "What kind of animals are walking on the street?"}]},
                            {"role": "assistant", "content": '["animal"]'},
                            {"role": "user", "content": [{"type": "text", "text": "How many clouds are in the sky above the houses?"}]},
                            {"role": "assistant", "content": '["cloud"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Do you notice any stars or the moon in the night sky?"}]},
                            {"role": "assistant", "content": '["star", "moon"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Is there a river flowing beside the mountains?"}]},
                            {"role": "assistant", "content": '["river"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Can you see a cat or dog in this photo?"}]},
                            {"role": "assistant", "content": '["cat", "dog"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Are there any bookshelves in the background?"}]},
                            {"role": "assistant", "content": '["bookshelf"]'},
                            {"role": "user", "content": [{"type": "text", "text": "What objects are visible in this image?"}]},
                            {"role": "assistant", "content": '["object"]'},
                            {"role": "user", "content": [{"type": "text", "text": "Are there any bottles and glasses on the table?"}]},
                            {"role": "assistant", "content": '["bottle", "glass"]'},
                        ]

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


prompt_describe_image = "Please provide a brief description of the provided image, focusing on elements relevant to energy usage, sources, and infrastructure. Highlight key features such as renewable energy installations, fossil fuel operations, energy consumption patterns, and any visible impacts on the environment. The description should be succinct and detailed enough to facilitate further analysis on energy forecasting, trading, and resource management. Include any observable contrasts between sustainable practices and traditional energy methods."

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


import json

def read_stream(stream):
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_data = json.loads(chunk.get("bytes", '{}'))
            if chunk_data.get("type") == "content_block_delta":
                # Print the response text as it streams
                print(chunk_data["delta"]["text"], end="", flush=True)
    else:
        print("No stream available. Please check the model response.")

from vision import numpy_array_to_base64
import numpy as np
import os
from PIL import Image

def format_prompt_describe_images(labels, prompt, results):
    return f"""You are provided with an image on which there are {labels}. Please answer the question: "{prompt}", focusing on elements relevant to an energy trader. Highlight key features and any visible potential impacts. The description should be succinct and detailed enough to facilitate further analysis on energy forecasting, trading, and resource management. The objects on the image are provided in a list containing the label and the box placement which are relevant to you: {results}. Write the response without using any markdown formatting, including asterisks or bold text."""

def format_message_describe_image(image,results,labels,prompt,history=None,metadata=None):
    
    base64_image = numpy_array_to_base64(image)

    list_image_examples = []

    folder = "examples"
    for filename in os.listdir(folder):
        image = Image.open(os.path.join(folder,filename))
        image_array = np.array(image) 
        list_image_examples.append(numpy_array_to_base64(image_array))

    messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": format_prompt_describe_images(["boat"],"How many boats are on the image, what are they doing?", [{'score': 0.4049810767173767, 'label': ['boat'], 'box': {'xmin': 370, 'ymin': 277.2151898734177, 'xmax': 388, 'ymax': 284.8101265822785}}, {'score': 0.18504545092582703, 'label': ['boat'], 'box': {'xmin': 714, 'ymin': 159.49367088607593, 'xmax': 737, 'ymax': 167.0886075949367}}])},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{list_image_examples[0]}"}}
                    ]
                }, 
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": """From the provided labels and bounding boxes in the list:

                        1. Total Boats Identified: Based on the provided annotations, there are two boats detected in the image.

                        2. Activities and Locations:
                        - Boat 1: Located at bounding box coordinates `(xmin=370, ymin=277.2, xmax=388, ymax=284.8)` with a confidence score of 0.40. This boat appears to be moving or docked close to the foreground of the scene.
                        - Boat 2: Located at bounding box coordinates `(xmin=714, ymin=159.5, xmax=737, ymax=167.1)` with a lower confidence score of 0.18. This boat is likely farther in the background and may be stationary or moving across the water.

                        3. Key Observations:
                        - One large cruise ship dominates the scene and is prominently featured near the center-right of the image. It may influence traffic flow or impact smaller boats in the vicinity.
                        - Smaller boats and potentially ferries or transport vessels are scattered throughout, suggesting activity typical of a busy urban waterway.
                        - The visible structures (e.g., docks, bridges) imply an area designed for heavy boat traffic.

                        4. Potential Impacts:
                        - Congestion: The large cruise ship may limit movement for smaller vessels.
                        - Environmental: Presence of large ships could contribute to waterway pollution or noise affecting nearby urban zones.
                        - Safety: Coordination among the various types of boats (smaller vs. larger vessels) may require monitoring to avoid collisions."""}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": format_prompt_describe_images(["car"],"What is happening with the cars on the image?", [{'score': 0.35145682096481323, 'label': ['car'], 'box': {'xmin': 602, 'ymin': 537.2239747634069, 'xmax': 666, 'ymax': 589.1692954784437}}, {'score': 0.21112746000289917, 'label': ['car'], 'box': {'xmin': 803, 'ymin': 546.7928496319663, 'xmax': 848, 'ymax': 598.7381703470031}}, {'score': 0.14581221342086792, 'label': ['car'], 'box': {'xmin': 691, 'ymin': 572.7655099894847, 'xmax': 785, 'ymax': 647.9495268138801}}])},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{list_image_examples[1]}"}}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": """The image depicts vehicles in a tunnel, which appears to be dimly lit with artificial lighting. Key observations:

                        1. Traffic Flow: The cars are moving within a confined space, indicating a controlled traffic flow.
                        2. Lighting Conditions: The lighting inside the tunnel is artificial, potentially affecting visibility for drivers.
                        3. Tunnel Structure: The tunnel walls and ceiling are reinforced, suggesting it is designed for heavy traffic loads or a long stretch.
                        4. Potential Impacts:
                        - Limited visibility could increase the risk of accidents.
                        - Traffic congestion is possible due to the confined space.
                        - Ventilation might be a concern if the tunnel lacks proper air circulation.

                        These factors could influence traffic management, safety measures, and structural maintenance requirements for the tunnel.
                        """}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": format_prompt_describe_images(labels, prompt, results)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                },
             ]
    
    if history:
        messages.extend(history)

    return messages