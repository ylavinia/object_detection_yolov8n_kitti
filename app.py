# app.py to deploy to Hugging Face Gradio

from ultralytics import YOLO
import gradio as gr
from PIL import Image
import numpy as np

# Load the trained YOLOv8n model
model = YOLO("best.pt")

def detect_objects(image, conf_threshold):
    # Run YOLO prediction
    results = model.predict(
        image,
        save=False,
        conf=conf_threshold
    )
    
    # Draw bounding boxes
    output_image = results[0].plot()

    # Convert numpy array → PIL
    output_pil = Image.fromarray(output_image)
    return output_pil

# Gradio Interface
demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label="Confidence Threshold")
    ],
    outputs=gr.Image(type="pil", label="Detected Objects"),
    title="YOLOv8n Object Detection on KITTI",
    description=(
        "Upload an image to detect vehicles, pedestrians, and cyclists using YOLOv8n. "
        "Adjust the confidence threshold for more or fewer detections."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
