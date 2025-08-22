import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import torch
import easyocr

# Streamlit app title
st.title("YOLO License Plate Detection (Image & Video)")

# Detect device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device=="cuda" else ""))

# OCR Engine (only EasyOCR)
reader = easyocr.Reader(['en'], gpu=(device=="cuda"))

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

# Demo image path in repo root
demo_image_path = "demo_image.jpg"  # Replace with your demo image filename

# Load YOLO model
try:
    model = YOLO("best.pt")  # Replace with your YOLO model
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

# OCR helper function
def run_ocr(roi):
    results = reader.readtext(roi)
    texts = [res[1] for res in results if res[1].strip() != ""]
    return " ".join(texts).strip()

# Image processing
def predict_and_save_image(path, output_path):
    try:
        results = model.predict(path, device=device)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detected_texts = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = image[y1:y2, x1:x2]
                text = run_ocr(roi)
                if text:
                    detected_texts.append(text)
                    cv2.putText(image, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image)
        return output_path, detected_texts
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, []

# Video processing
def predict_and_plot_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None, []

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # safer codec for Streamlit
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        detected_texts = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, device=device)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]
                    text = run_ocr(roi)
                    if text:
                        detected_texts.append(text)
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        return output_path, detected_texts
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, []

# Handle uploaded file
if uploaded_file is not None:
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", f"output_{uploaded_file.name}")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("Processing...")

    if input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        result_path, texts = predict_and_plot_video(input_path, output_path)
        if result_path:
            st.success("Video processed successfully!")
            st.video(result_path)
            if texts:
                st.subheader("Detected License Plate Numbers:")
                st.write(list(set(texts)))
    else:
        result_path, texts = predict_and_save_image(input_path, output_path)
        if result_path:
            st.image(Image.open(result_path))
            if texts:
                st.subheader("Detected License Plate Numbers:")
                st.write(list(set(texts)))

# If no file uploaded, process demo image
else:
    if os.path.exists(demo_image_path):
        st.info("No file uploaded. Showing processed demo image.")
        output_demo_path = os.path.join("temp", "output_demo_image.jpg")
        result_path, texts = predict_and_save_image(demo_image_path, output_demo_path)
        if result_path:
            st.image(Image.open(result_path))
            if texts:
                st.subheader("Detected License Plate Numbers:")
                st.write(list(set(texts)))
    else:
        st.error("Demo image not found. Please add a demo image to the repo root.")
