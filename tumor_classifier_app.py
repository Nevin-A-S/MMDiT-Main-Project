import streamlit as st
import torch
from PIL import Image
import os
from pathlib import Path
import numpy as np
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig

import sys
sys.path.append(".")
from visionTransformer.vitTrain import ViTFineTuner

st.set_page_config(
    page_title="Tumor Classification",
    page_icon="🔬",
    layout="wide"
)

@st.cache_resource
def load_model(checkpoint_path, model_name="google/vit-base-patch16-224-in21k", num_classes=None):
    """
    Load the ViT model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_name: Name of the pre-trained model
        num_classes: Number of classes for classification
        
    Returns:
        model: Loaded model
        idx_to_label: Dictionary mapping from index to label
    """
    # Initialize the ViTFineTuner
    fine_tuner = ViTFineTuner(
        model_name=model_name,
        num_classes=num_classes,
        mixed_precision=False  # Disable mixed precision for inference
    )
    
    # Setup the model
    fine_tuner.setup_model(num_classes=num_classes)
    
    # Load the model checkpoint
    fine_tuner.load_model(checkpoint_path)
    
    # Get the idx_to_label mapping if available
    idx_to_label = getattr(fine_tuner, 'idx_to_label', None)
    if idx_to_label is None:
        # If not available, create a default mapping
        idx_to_label = {i: f"Class {i}" for i in range(num_classes)}
    
    return fine_tuner, idx_to_label

def preprocess_image(image):
    """
    Preprocess the image for the ViT model.
    
    Args:
        image: PIL Image
        
    Returns:
        tensor: Preprocessed image tensor
    """
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])
    
    return transform(image).unsqueeze(0)  

def predict(fine_tuner, image_tensor, idx_to_label):
    """
    Make predictions on an image.
    
    Args:
        fine_tuner: ViTFineTuner instance
        image_tensor: Preprocessed image tensor
        idx_to_label: Dictionary mapping from index to label
        
    Returns:
        prediction: Predicted class
        confidence: Prediction confidence
    """
    device = fine_tuner.device
    model = fine_tuner.model
    
    image_tensor = image_tensor.to(device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    predicted_label = idx_to_label[predicted_class.item()]
    
    return predicted_label, confidence.item()

def main():
    st.title("Tumor Classification")
    st.write("Upload an image to classify the type of tumor")
    
    st.sidebar.title("Model Configuration")
    
    checkpoint_dir = "vit_checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
    else:
        checkpoint_files = []
    
    if not checkpoint_files:
        st.error("No model checkpoints found in the 'vit_checkpoints' directory. Please train a model first.")
        return
    
    selected_checkpoint = st.sidebar.selectbox(
        "Select Model Checkpoint",
        checkpoint_files
    )
    
    num_classes = st.sidebar.number_input("Number of Classes", min_value=2, value=10)
    
    checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
    
    try:
        fine_tuner, idx_to_label = load_model(checkpoint_path, num_classes=num_classes)
        st.sidebar.success(f"Model loaded successfully: {selected_checkpoint}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return
    
    st.sidebar.subheader("Class Mapping")
    for idx, label in idx_to_label.items():
        st.sidebar.write(f"Class {idx}: {label}")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        image_tensor = preprocess_image(image)
        
        with st.spinner("Classifying..."):
            predicted_label, confidence = predict(fine_tuner, image_tensor, idx_to_label)
        
        with col2:
            st.subheader("Prediction Results")
            st.write(f"**Predicted Tumor Type:** {predicted_label}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            st.progress(confidence)
            
            if confidence < 0.5:
                st.warning("Low confidence prediction. Consider using a different image or model.")
            elif confidence > 0.8:
                st.success("High confidence prediction.")

if __name__ == "__main__":
    main() 