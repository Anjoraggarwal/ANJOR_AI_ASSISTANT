import os
import time
import gradio as gr
from dotenv import load_dotenv

from speech_to_text import transcribe_with_groq
from ai_agent import ask_agent
from text_to_speech import text_to_speech_with_elevenlabs

load_dotenv()

def agent_voice_chat(audio_file, chat_history):
    if not audio_file:
        chat_history = chat_history or []
        chat_history.append({"role": "assistant", "content": "⚠️ No audio received. Please record and try again."})
        return chat_history, None

    user_input = transcribe_with_groq(audio_file)
    response = ask_agent(user_input)
    tts_filename = f"final_{int(time.time())}.mp3"
    text_to_speech_with_elevenlabs(response, tts_filename)
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})
    return chat_history, tts_filename


# -----------------------------------------------------------------
# -------- Webcam code, preserved from your original setup ---------
import cv2

camera = None
is_running = False
last_frame = None

def initialize_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera is not None and camera.isOpened()

def start_webcam():
    global is_running, last_frame
    is_running = True
    if not initialize_camera():
        return None
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        return frame
    return last_frame

def stop_webcam():
    global is_running, camera
    is_running = False
    if camera is not None:
        camera.release()
        camera = None
    return None

def get_webcam_frame():
    global camera, is_running, last_frame
    if not is_running or camera is None:
        return last_frame
    if camera.get(cv2.CAP_PROP_BUFFERSIZE) > 1:
        for _ in range(int(camera.get(cv2.CAP_PROP_BUFFERSIZE)) - 1):
            camera.read()
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        return frame
    return last_frame

# -----------------------------------------------------------------
# ------------------------ Gradio UI Setup ------------------------
with gr.Blocks(css=""" /* You can keep your custom CSS, omitted for brevity */ """) as demo:
    gr.HTML("<h1 style='color:#fff;text-align:center;'>Anjor AI Assistant</h1>")
    with gr.Row():
        # Webcam
        with gr.Column():
            gr.Markdown("### Webcam Feed")
            with gr.Row():
                start_btn = gr.Button("🎥 Start Camera")
                stop_btn = gr.Button("🛑 Stop Camera")
            webcam_output = gr.Image(
                label="Live Feed", streaming=True, show_label=False, width=640, height=480
            )
            webcam_timer = gr.Timer(0.033)  # ~30 FPS

        # Chat/Audio
        with gr.Column():
            gr.Markdown("### Voice Chat with Anjor")
            mic = gr.Audio(sources="microphone", type="filepath", label="🎤 Speak to Anjor")
            ask_btn = gr.Button("Ask")
            chatbot = gr.Chatbot(label="Conversation", height=400, show_label=False, type="messages")
            audio_out = gr.Audio(label="Anjor Speaks")

            clear_btn = gr.Button("🧹 Clear Chat")

    # Webcam controls
    start_btn.click(fn=start_webcam, outputs=webcam_output)
    stop_btn.click(fn=stop_webcam, outputs=webcam_output)
    webcam_timer.tick(fn=get_webcam_frame, outputs=webcam_output, show_progress=False)

    # Voice QA controls
    ask_btn.click(
        fn=agent_voice_chat,
        inputs=[mic, chatbot],
        outputs=[chatbot, audio_out]
    )
    clear_btn.click(fn=lambda: [], outputs=chatbot)

# -----------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True
    )
