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
