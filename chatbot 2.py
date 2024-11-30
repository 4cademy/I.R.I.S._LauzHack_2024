import boto3
import json
import os

# Initialize the Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.environ.get('AWS_SESSION_TOKEN')
)

def define_kwargs(model_id, prompt):
    """
    Constructs the request parameters for invoking the Bedrock model.

    Parameters:
    - model_id (str): The identifier of the model to be used.
    - prompt (str): The input prompt for the model.

    Returns:
    - dict: The request parameters to invoke the Bedrock model.
    """
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
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        })
    }

def stream_bedrock_response(prompt, model_id="anthropic.claude-3-5-sonnet-20241022-v2:0"):
    """
    Sends a prompt to a Bedrock model and streams the response.

    Parameters:
    - prompt (str): The input prompt to send to the model.
    - model_id (str): The model ID to use for the request (default is Claude 3.5).

    Returns:
    - None: Outputs the response stream to the console in real-time.
    """
    kwargs = define_kwargs(model_id, prompt)

    try:
        # Invoke the model and retrieve the response stream
        response = bedrock_runtime.invoke_model_with_response_stream(**kwargs)
        stream = response.get("body")

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
    except Exception as e:
        print(f"An error occurred while streaming the response: {e}")

# def analyze_depending_on_...

# Example prompt
if __name__ == "__main__":
    user_prompt = "Hey, how can I help you today?"
    stream_bedrock_response(user_prompt)
