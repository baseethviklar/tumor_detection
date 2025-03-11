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
CORS(app)

# Load the models only when needed (Lazy Loading)
DETECTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vit_brain_tumor_detection.pth")
SEGMENTATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "tumor_segmentation.pth")

# Force use CPU since Render free tier has no GPU
device = torch.device("cpu")

def load_detection_model():
    model = ViTForBrainTumorDetection()
    model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_segmentation_model():
    model = TumorSegmentationModel()
    model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device), image


def generate_segmentation_mask(image_tensor, original_image, segmentation_model):
    with torch.no_grad():
        segmentation_output = segmentation_model(image_tensor)
    segmentation_mask = torch.sigmoid(segmentation_output).squeeze().cpu().numpy()
    segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
    segmentation_mask = cv2.resize(segmentation_mask, original_image.size)
    return segmentation_mask


def create_overlay_image(original_image, segmentation_mask):
    original_cv = np.array(original_image)
    original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(original_cv)
    overlay[:, :, 2] = segmentation_mask
    highlighted_image = cv2.addWeighted(original_cv, 1, overlay, 0.5, 0)
    highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)
    highlighted_pil = Image.fromarray(highlighted_image)
    buffer = io.BytesIO()
    highlighted_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.route('/api/detect-tumor', methods=['POST'])
def detect_tumor():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        image_tensor, original_image = preprocess_image(image_bytes)

        # Load models lazily
        detection_model = load_detection_model()
        segmentation_model = load_segmentation_model()

        # Predict tumor
        with torch.no_grad():
            outputs = detection_model(image_tensor)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence = probabilities[0][1].item() * 100
        has_tumor = confidence > 50

        # Free up GPU memory
        del image_tensor
        torch.cuda.empty_cache()

        result = {
            'hasTumor': bool(has_tumor),
            'confidence': float(confidence)
        }

        if has_tumor:
            segmentation_mask = generate_segmentation_mask(image_tensor, original_image, segmentation_model)
            overlay_image = create_overlay_image(original_image, segmentation_mask)
            result['overlayImage'] = f"data:image/png;base64,{overlay_image}"

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
