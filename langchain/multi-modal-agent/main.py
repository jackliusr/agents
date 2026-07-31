
from dotenv import load_dotenv
load_dotenv()



#### Model Initialization
from langchain_core.messages import HumanMessage

from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3.6:35b-a3b-q4_K_M",
    temperature=0,
    # other params...
)

import base64
import httpx

img_url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Nature_Boardwalk_Lincoln_Park.JPG"
def get_img_data(url):
    with httpx.Client() as client:
        response = client.get(img_url)
        response.raise_for_status()

        image_bytes = response.content
        content_type = response.headers.get("Content-Type", "image/png")
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        return image_data
def get_img_data2(img):
    with open(img, "rb") as image_file:
        # Read the file and encode it to base64 bytes
        base64_bytes = base64.b64encode(image_file.read())
        
        # Decode bytes to a clean UTF-8 string
        base64_string = base64_bytes.decode("utf-8")
        return base64_string

image_data = get_img_data2("./image.jpg")

message = HumanMessage(
    content=[
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)


response = model.invoke([message])
print(response.content)
