# app.py - Flask backend for brain tumor detection
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
from models.vit_model import ViTForBrainTumorDetection
from models.segmentation_model import TumorSegmentationModel

app = Flask(__name__)
CORS(app) # Enable CORS for cross-origin requests from React frontend

# Load the pre-trained models with relative paths
DETECTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vit_brain_tumor_detection.pth")
SEGMENTATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "tumor_segmentation.pth")

# Initialize the models
device = torch.device("cpu")  # Force CPU for Render compatibility

try:
    detection_model = ViTForBrainTumorDetection()
    detection_model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location=device))
    detection_model.to(device)
    detection_model.eval()

    segmentation_model = TumorSegmentationModel()
    segmentation_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=device))
    segmentation_model.to(device)
    segmentation_model.eval()
except Exception as e:
    print(f"Error loading models: {e}")

# Image preprocessing for ViT
def preprocess_image(image_bytes):
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Preprocessing for ViT model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device), image

# Generate segmentation mask
def generate_segmentation_mask(image_tensor, original_image):
    # Get segmentation prediction
    with torch.no_grad():
        segmentation_output = segmentation_model(image_tensor)
    # Process the segmentation output to get a binary mask
    segmentation_mask = torch.sigmoid(segmentation_output).squeeze().cpu().numpy()
    segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
    # Resize mask to original image dimensions
    original_size = original_image.size
    segmentation_mask = cv2.resize(segmentation_mask, original_size)
    return segmentation_mask

# Create highlighted image with tumor overlay
def create_overlay_image(original_image, segmentation_mask):
    # Convert PIL image to OpenCV format
    original_cv = np.array(original_image)
    original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)
    # Create red overlay for tumor regions
    overlay = np.zeros_like(original_cv)
    overlay[:, :, 2] = segmentation_mask # Red channel
    # Blend images
    alpha = 0.5
    highlighted_image = cv2.addWeighted(original_cv, 1, overlay, alpha, 0)
    # Convert back to PIL for encoding
    highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)
    highlighted_pil = Image.fromarray(highlighted_image)
    # Encode as base64
    buffer = io.BytesIO()
    highlighted_pil.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

@app.route('/api/detect-tumor', methods=['POST'])
def detect_tumor():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        # Preprocess image
        image_tensor, original_image = preprocess_image(image_bytes)
        # Get tumor detection prediction
        with torch.no_grad():
            outputs = detection_model(image_tensor)
        # Assuming outputs has logits and confidence
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence = probabilities[0][1].item() * 100 # Assuming class 1 is tumor
        has_tumor = confidence > 50
        result = {
            'hasTumor': bool(has_tumor),
            'confidence': float(confidence)
        }
        # If tumor detected, generate segmentation mask
        if has_tumor:
            segmentation_mask = generate_segmentation_mask(image_tensor, original_image)
            overlay_image = create_overlay_image(original_image, segmentation_mask)
            result['overlayImage'] = overlay_image
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
