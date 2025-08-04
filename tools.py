import cv2
import base64
from groq import Groq

def capture_image() -> str:
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            for _ in range(10):  # Warm up
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            _, buf = cv2.imencode('.jpg', frame)
            if _:
                return base64.b64encode(buf).decode('utf-8')
    raise RuntimeError("Could not open any webcam (tried indices 0-3)")

def analyze_image_with_query(query: str) -> str:
    """
    Takes a query and answers using a vision model by capturing a live webcam photo if the question requires.
    """

    img_b64 = capture_image()
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    if not query or not img_b64:
        return "Error: both 'query' and 'image' fields required."
    client = Groq()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message.content

