import streamlit as st
import torch
from PIL import Image
import os
from pathlib import Path
import numpy as np
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
import json
import numpy

import sys
sys.path.append(".")
from visionTransformer.vitTrain import ViTFineTuner, LabelEncoder

st.set_page_config(
    page_title="MRI Disease Classification",
    page_icon="🔬",
    layout="wide"
)

# Add safe globals for NumPy objects to handle PyTorch 2.6+ security restrictions
try:
    from torch.serialization import add_safe_globals
    # Add numpy scalar to safe globals
    add_safe_globals([numpy._core.multiarray.scalar])
except (ImportError, AttributeError):
    # Older PyTorch versions don't have this
    pass

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
        label_encoder: Label encoder for class mapping
    """
    # Check if label encoder file exists
    label_encoder_path = checkpoint_path.replace('.pth', '_label_encoder.json')
    label_encoder = None
    
    if os.path.exists(label_encoder_path):

        print(f"Loading label encoder from {label_encoder_path}")   

        try:
            label_encoder = LabelEncoder().load(label_encoder_path)
            st.sidebar.success(f"Label encoder loaded from {os.path.basename(label_encoder_path)}")
            # If num_classes wasn't specified, get it from the label encoder
            if num_classes is None:
                num_classes = label_encoder.num_classes
        except Exception as e:
            st.sidebar.warning(f"Error loading label encoder: {str(e)}")
    
    # If num_classes is still None, use a default value
    if num_classes is None:
        num_classes = 3  # Default to 3 classes
        st.sidebar.warning(f"Using default number of classes: {num_classes}")
    
    # Initialize the ViTFineTuner
    fine_tuner = ViTFineTuner(
        model_name=model_name,
        num_classes=num_classes,
        mixed_precision=False,  # Disable mixed precision for inference
        use_wandb=False  # Disable wandb for inference
    )
    
    # Setup the model
    fine_tuner.setup_model(num_classes=num_classes)
    
    # Load the model checkpoint
    try:
        # Override the load_checkpoint method to handle PyTorch 2.6+ security restrictions
        def safe_load_checkpoint(path):
            print(f"Loading checkpoint from {path}")
            try:
                # First try with weights_only=False (less secure but compatible with older checkpoints)
                checkpoint = torch.load(path, map_location=fine_tuner.device, weights_only=False)
            except (TypeError, ValueError):
                # For older PyTorch versions that don't have weights_only parameter
                checkpoint = torch.load(path, map_location=fine_tuner.device)
            
            # Load model state
            fine_tuner.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            fine_tuner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load EMA state if available
            if fine_tuner.use_ema and 'ema_state_dict' in checkpoint:
                for name, param in checkpoint['ema_state_dict'].items():
                    fine_tuner.ema_model.ema[name] = param.to(fine_tuner.device)
            
            return checkpoint
        
        # Replace the method with our safe version
        fine_tuner.load_checkpoint = safe_load_checkpoint
        
        # Now load the checkpoint
        checkpoint = fine_tuner.load_checkpoint(checkpoint_path)
        st.sidebar.success(f"Model checkpoint loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Error loading model checkpoint: {str(e)}")
        raise e
    
    return fine_tuner, label_encoder

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

def predict(fine_tuner, image_tensor, label_encoder=None):
    """
    Make predictions on an image.
    
    Args:
        fine_tuner: ViTFineTuner instance
        image_tensor: Preprocessed image tensor
        label_encoder: Label encoder for class mapping
        
    Returns:
        prediction: Predicted class
        confidence: Prediction confidence
        all_probs: All class probabilities
    """
    device = fine_tuner.device
    model = fine_tuner.model
    
    image_tensor = image_tensor.to(device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    # Convert to numpy for easier handling
    all_probs = probabilities[0].cpu().numpy()
    predicted_idx = predicted_class.item()
    
    # Convert predicted index to label if label encoder is available
    if label_encoder is not None:
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    else:
        predicted_label = f"Class {predicted_idx}"
    
    return predicted_label, confidence.item(), all_probs

def main():
    st.title("MRI Disease Classification")
    st.write("Upload an image to classify the MRI")
    
    # Sidebar for model selection
    st.sidebar.title("Model Configuration")
    
    # Get available checkpoint files
    checkpoint_dirs = ["checkpoints", "vit_checkpoints"]
    checkpoint_files = []
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pth') and not f.endswith('_label_encoder.json'):
                    checkpoint_files.append(os.path.join(checkpoint_dir, f))
    
    if not checkpoint_files:
        st.error("No model checkpoints found in the 'checkpoints' or 'vit_checkpoints' directories. Please train a model first.")
        return
    
    # Model selection
    selected_checkpoint = st.sidebar.selectbox(
        "Select Model Checkpoint",
        checkpoint_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Load the model
    try:
        fine_tuner, label_encoder = load_model(selected_checkpoint)
        st.sidebar.success(f"Model loaded successfully: {os.path.basename(selected_checkpoint)}")
        
        # Display class mapping if available
        if label_encoder is not None:
            st.sidebar.subheader("Class Mapping")
            for idx, label in label_encoder.idx_to_label.items():
                st.sidebar.write(f"Class {idx}: {label}")
        else:
            st.sidebar.warning("No label encoder found. Class indices will be displayed instead of labels.")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        image_tensor = preprocess_image(image)
        
        with st.spinner("Classifying..."):
            predicted_label, confidence, all_probs = predict(fine_tuner, image_tensor, label_encoder)
        
        with col2:
            st.subheader("Prediction Results")
            st.write(f"**Predicted Type:** {predicted_label}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            st.progress(confidence)
            
            if confidence < 0.1:
                st.warning("Low confidence prediction. Consider using a different image or model.")
            elif confidence > 0.8:
                st.success("High confidence prediction.")
            
            # Display all class probabilities
            st.subheader("All Class Probabilities")
            
            if label_encoder is not None:
                for idx, prob in enumerate(all_probs):
                    label = label_encoder.inverse_transform([idx])[0]
                    st.write(f"{label}: {prob:.2%}")
                    st.progress(float(prob))
            else:
                for idx, prob in enumerate(all_probs):
                    st.write(f"Class {idx}: {prob:.2%}")
                    st.progress(float(prob))

if __name__ == "__main__":
    main() 