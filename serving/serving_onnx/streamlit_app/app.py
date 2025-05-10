import os
import cv2
import streamlit as st
import tempfile
from pathlib import Path
from ultralytics import YOLO
import time
from PIL import Image
import boto3
from datetime import datetime
from mimetypes import guess_type
import uuid
from concurrent.futures import ThreadPoolExecutor

# Configuration from environment variables
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "triton_server:8000")
MODEL_NAME = os.getenv("CHEST_XRAY_MODEL_NAME", "chest_xray_detector")
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "your-access-key")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "your-secret-key")
BUCKET_NAME = "production"

# Initialize MinIO client
s3 = boto3.client(
    's3',
    endpoint_url=MINIO_URL,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name='us-east-1'
)

# Thread pool for asynchronous uploads
executor = ThreadPoolExecutor(max_workers=2)

def upload_to_minio(img_path, predictions, confidences, prediction_id):
    """Upload image to MinIO with metadata tags."""
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    predicted_class = predictions[0] if predictions else "Unknown"  # Use the first predicted class
    confidence = confidences[0] if confidences else 0.0
    class_dir = f"class_{predicted_class.replace(' ', '_')}"

    # Get file extension and content type
    _, ext = os.path.splitext(img_path)
    content_type = guess_type(img_path)[0] or 'application/octet-stream'
    s3_key = f"{class_dir}/{prediction_id}{ext}"

    # Upload the image
    with open(img_path, 'rb') as f:
        s3.upload_fileobj(
            f,
            BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': content_type}
        )

    # Add tags (predicted class, confidence, timestamp)
    s3.put_object_tagging(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Tagging={
            'TagSet': [
                {'Key': 'predicted_class', 'Value': predicted_class},
                {'Key': 'confidence', 'Value': f"{confidence:.3f}"},
                {'Key': 'timestamp', 'Value': timestamp}
            ]
        }
    )

st.title("Chest X-Ray Detection using YOLOV11")

# Server URL and model name input
server_url = st.text_input("Triton Server URL", value=TRITON_SERVER_URL)
model_name = st.text_input("Model Name", value=MODEL_NAME)

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    # Process image when button is clicked
    if st.button("Detect"):
        try:
            with st.spinner("Running inference..."):
                st.info(f"Connecting to Triton Server at: http://{server_url}/{model_name}")
                
                # Load the Triton Server model
                model = YOLO(f"http://{server_url}/{model_name}", task="detect")
                
                # Run inference
                start_time = time.time()
                results = model(temp_path)
                inference_time = time.time() - start_time
                
                # Display results
                st.success(f"Detection completed in {inference_time:.4f} seconds")
                
                # Extract predictions and confidences
                predictions = []
                confidences = []
                for result in results:
                    # Using the built-in plotting from YOLO results
                    result_img = result.plot()
                    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                    st.image(result_img_bgr, caption="Detection Results")
                    
                    # Show the detection details
                    boxes = result.boxes
                    if len(boxes) > 0:
                        st.write(f"Found {len(boxes)} detections:")
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = result.names[cls]
                            predictions.append(label)
                            confidences.append(conf)
                            st.write(f"- Detection {i+1}: {label} (Confidence: {conf:.3f})")
                    else:
                        st.write("No detections found.")
                        predictions.append("None")
                        confidences.append(0.0)
                
                # Save to MinIO asynchronously
                prediction_id = str(uuid.uuid4())
                executor.submit(upload_to_minio, temp_path, predictions, confidences, prediction_id)
        
        except Exception as e:
            import traceback
            st.error(f"Error: {e} \n {traceback.format_exc()}")
            st.warning("Make sure the Triton server is running and the model is loaded correctly.")
    
    # Clean up the temporary file
    os.unlink(temp_path)

# Footer with information
st.markdown("---")
st.caption("Using YOLOV11 model deployed on NVIDIA Triton Inference Server")