import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import os
import torch
import easyocr

# App title
st.title("YOLO License Plate Detection")

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device=="cuda" else ""))

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=(device=="cuda"))

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

# Default demo image
demo_image_path = "demo_image.jpg"  # place your demo image in root

# Load YOLO model
try:
    model = YOLO("best.pt")  # Replace with your trained YOLO model
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

# OCR helper
def run_ocr(roi):
    results = reader.readtext(roi)
    texts = [res[1].strip() for res in results if res[1].strip() != ""]
    return " ".join(texts)

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
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                roi = image[y1:y2, x1:x2]
                text = run_ocr(roi)
                if text:
                    detected_texts.append(text)
                    cv2.putText(image, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return output_path, list(set(detected_texts))

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, []

# Video processing (no preview, only download)
def predict_and_save_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None, []

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    roi = frame[y1:y2, x1:x2]
                    text = run_ocr(roi)
                    if text:
                        detected_texts.append(text)
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        return output_path, list(set(detected_texts))

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, []

# Main logic
if uploaded_file is not None:
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", f"output_{uploaded_file.name}")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
else:
    # Use demo image if no upload
    input_path = demo_image_path
    output_path = os.path.join("temp", f"output_demo.jpg")

# Check type and process
if input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
    result_path, texts = predict_and_save_video(input_path, output_path)
    if result_path:
        st.success("Video processed! Download below:")
        st.download_button(
            label="Download Processed Video",
            data=open(result_path, "rb"),
            file_name="output_video.mp4",
            mime="video/mp4"
        )
        if texts:
            st.subheader("Detected License Plate Numbers:")
            st.write(texts)
else:
    result_path, texts = predict_and_save_image(input_path, output_path)
    if result_path:
        st.image(Image.open(result_path))
        if texts:
            st.subheader("Detected License Plate Numbers:")
            st.write(texts)

