import gradio as gr
import numpy as np
import time
from serpapi import GoogleSearch
from vision import invoke_owlv2_endpoint, annotate_image
from chatbot import stream_bedrock_response
import json
import cv2

# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values

# loading variables from .env file
load_dotenv()

# getting the value of the API_KEY variable
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")


def start():
    r_list = [gr.Image("images/logo.jpg", scale=1, show_label=False, height=200, interactive=False, visible=True,
                       container=False, show_download_button=False, show_share_button=False,
                       show_fullscreen_button=False),
              gr.Button(visible=False),
              gr.HTML("<h1>Which images shall I find?</h1>", visible=True),
              gr.Textbox(lines=1, container=False, placeholder="Type here...", visible=True),
              gr.Button("Search", visible=True)]

    for i in range(0, 6):
        r_list.append(gr.Image(scale=1, show_label=False, height=200, interactive=False, visible=True))

    return r_list


def search_images(query):
    params = {
        "q": query,
        "engine": "google_images",
        "api_key": SEARCH_API_KEY,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    links = []
    for result in results["images_results"]:
        link = result["thumbnail"]
        website = result["link"]
        if (not website.startswith("https://www.alamy.com")
                and not website.startswith("https://www.istockphoto.com")
                and not website.startswith("https://www.mdpi.com")
                and not website.startswith("https://www.researchgate.net")
                and not website.startswith("https://www.gettyimages.com")
                and not website.startswith("https://www.shutterstock.com")):
            links.append(link)
            print(website)
        if len(links) == 6:
            break

    r_list = []
    for i in range(0, 6):
        r_list.append(gr.Image(links[i], scale=1, show_label=False, height=200, interactive=False, visible=True))
    for i in range(0, 6):
        r_list.append(gr.Button("Use this image", visible=True))

    r_list.append(gr.HTML("<h1>What shall I analyze in the image?</h1>", visible=True))
    r_list.append(gr.Textbox(lines=1, container=False, placeholder="Enter question...", visible=True))
    r_list.append(gr.Button("Ask", visible=True))
    r_list.append(gr.ImageEditor(
        show_label=False,
        sources=["upload", "clipboard"],
        type="numpy",
        visible=True,
    ))
    r_list.append(gr.HTML("<hr>", visible=True))
    return r_list


def move_image(im):
    desired_width = 1000
    desired_height = 1000
    height, width, channels = im.shape

    if width < desired_width:
        desired_height = int(desired_width * height / width)

    resized_im = cv2.resize(im, (desired_width, desired_height), interpolation=cv2.INTER_CUBIC)
    editor = gr.ImageEditor(
        value=resized_im,
        type="numpy",
        scale=2,
    )

    return editor


def analyze_image(im, promt):
    og_im = gr.Image(im, visible=False)

    # Show the new image instead as a np array (opencv)
    # TODO: Change what to search for
    results = invoke_owlv2_endpoint(im, [["car"]])
    result_state = results

    im = annotate_image(im, results, score_threshold=.1)

    r_list = [im, gr.HTML("<hr>", visible=True),
              gr.Chatbot(type="messages", visible=True),
              gr.Textbox(scale=6, container=False, visible=True),
              gr.Button("Send", scale=1, visible=True),
              gr.Slider(minimum=0, maximum=1, step=0.01, value=0.1, label="Threshold", interactive=True, visible=True),
              gr.Button("Reanalyze", visible=True),
              result_state,
              gr.Textbox(scale=6, container=False, visible=False),
              gr.Button("Send", scale=1, visible=False),
              og_im]
    return r_list


def reanalyze_image(og_im, slider, result_state):
    im = annotate_image(og_im, result_state, score_threshold=slider)
    return im


def user(user_message, history: list):
    return "", history + [{"role": "user", "content": user_message}]


def bot(history: list):
    prompt = history[-1]['content']
    stream = stream_bedrock_response(prompt)

    history.append({"role": "assistant", "content": ""})

    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_data = json.loads(chunk.get("bytes", '{}'))
                if chunk_data.get("type") == "content_block_delta":
                    # Print the response text as it streams
                    history[-1]['content'] += chunk_data["delta"]["text"]
                    time.sleep(0.01)
                    yield history

    # bot_message is the actual message of the chatbot
    # bot_message = "How are you? This is a test message. Just to see how the chatbot looks. Testing 1, 2, 3."
    # history.append({"role": "assistant", "content": ""})
    # for character in bot_message:
    #     history[-1]['content'] += character
    #     time.sleep(0.01)
    #     yield history


with gr.Blocks() as demo:
    title_image = gr.Image("images/logo.jpg", scale=1, show_label=False, height=700, interactive=False, visible=True,
                           container=False, show_download_button=False, show_share_button=False,
                           show_fullscreen_button=False)
    start_btn = gr.Button("Launch I.R.I.S", visible=True)

    q1 = gr.HTML("<h1>Which images shall I find?</h1>", visible=False)
    search_textbox = gr.Textbox(lines=1, container=False, placeholder="Type here...", visible=False)
    search_btn = gr.Button("Search", visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            im1 = gr.Image(scale=1, show_label=False, height=200, interactive=False, visible=False)
            b1 = gr.Button("Use this image", visible=False)
        with gr.Column(scale=1):
            im2 = gr.Image(scale=1, show_label=False, height=200, interactive=False, visible=False)
            b2 = gr.Button("Use this image", visible=False)
        with gr.Column(scale=1):
            im3 = gr.Image(scale=1, show_label=False, height=200, interactive=False, visible=False)
            b3 = gr.Button("Use this image", visible=False)
    with gr.Row():
        with gr.Column(scale=1):
            im4 = gr.Image(scale=1, show_label=False, height=200, interactive=False, visible=False)
            b4 = gr.Button("Use this image", visible=False)
        with gr.Column(scale=1):
            im5 = gr.Image(scale=1, show_label=False, height=200, interactive=False, visible=False)
            b5 = gr.Button("Use this image", visible=False)
        with gr.Column(scale=1):
            im6 = gr.Image(scale=1, show_label=False, height=200, interactive=False, visible=False)
            b6 = gr.Button("Use this image", visible=False)

    line1 = gr.HTML("<hr>", visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            q2 = gr.HTML("<h1>What shall I analyze in the image?</h1>", visible=False)
            ask_textbox = gr.Textbox(lines=1, container=False, visible=False)
            ask_btn = gr.Button("Ask", visible=False)
            threshold_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.1, label="Threshold",
                                         interactive=True, visible=False)
            reanalyze_btn = gr.Button("Reanalyze", visible=False)

            im_result_state = gr.State()
        used_image = gr.Image(
            show_label=False,
            sources=["upload", "clipboard"],
            type="numpy",
            visible=False,
            scale=2,
        )
        original_image = gr.Image(
            show_label=False,
            sources=["upload", "clipboard"],
            type="numpy",
            visible=False,
            scale=2,
        )

    line2 = gr.HTML("<hr>", visible=False)

    chatbot = gr.Chatbot(type="messages", visible=False)
    with gr.Row():
        msg = gr.Textbox(scale=6, container=False, visible=False)
        send_btn = gr.Button("Send", scale=1, visible=False)

    # Button click events
    start_btn.click(fn=start,
                    outputs=[title_image, start_btn, q1, search_textbox, search_btn, im1, im2, im3, im4, im5, im6])

    search_btn.click(fn=search_images, inputs=search_textbox,
                     outputs=[im1, im2, im3, im4, im5, im6, b1, b2, b3, b4, b5, b6, q2, ask_textbox, ask_btn,
                              used_image,
                              line1])

    b1.click(fn=move_image, inputs=im1, outputs=used_image)
    b2.click(fn=move_image, inputs=im2, outputs=used_image)
    b3.click(fn=move_image, inputs=im3, outputs=used_image)
    b4.click(fn=move_image, inputs=im4, outputs=used_image)
    b5.click(fn=move_image, inputs=im5, outputs=used_image)
    b6.click(fn=move_image, inputs=im6, outputs=used_image)

    ask_btn.click(fn=analyze_image, inputs=[used_image, ask_textbox],
                  outputs=[used_image, line2, chatbot, msg, send_btn, threshold_slider, reanalyze_btn, im_result_state,
                           ask_textbox, ask_btn, original_image]).then(fn=user, inputs=[ask_textbox, chatbot],
                                                                       outputs=[msg, chatbot], queue=False).then(
        fn=bot, inputs=chatbot, outputs=chatbot
    )

    reanalyze_btn.click(fn=reanalyze_image, inputs=[original_image, threshold_slider, im_result_state],
                        outputs=used_image)  # .then(fn=user, inputs=[ask_textbox, chatbot],
    # outputs=[msg, chatbot], queue=False).then( fn=bot, inputs=chatbot, outputs=chatbot )

    send_btn.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=bot, inputs=chatbot, outputs=chatbot
    )
    msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        fn=bot, inputs=chatbot, outputs=chatbot
    )

if __name__ == "__main__":
    demo.launch()
