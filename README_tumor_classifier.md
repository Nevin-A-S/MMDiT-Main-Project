# Tumor Classification App

This Streamlit application allows you to classify tumor types using a Vision Transformer (ViT) model trained with the `vitTrain.py` script.

## Features

- Upload images to classify tumor types
- Select from available trained models
- View prediction confidence and results
- User-friendly interface with image preview
- Default mapping for common tumor types
- Option to customize class names for your specific model
- Automatic loading of label mappings saved during training

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Streamlit
- PIL (Pillow)

## Installation

1. Make sure you have all the required packages installed:

```bash
pip install streamlit torch torchvision transformers pillow
```

2. Ensure you have trained models available in the `vit_checkpoints` directory. If not, you need to train models using the `vitTrain.py` script first.

## Usage

1. Run the Streamlit application:

```bash
streamlit run tumor_classifier_app.py
```

2. The application will open in your default web browser.

3. In the sidebar, select a trained model checkpoint and specify the number of classes.

4. The application will automatically load any label mapping file associated with the selected model checkpoint. You have two options:
   - Use the loaded label mapping (recommended for consistency with training)
   - Create custom labels by unchecking "Use loaded label mapping"

5. Upload an image of a tumor using the file uploader.

6. The application will display the uploaded image and the prediction results, including the predicted tumor type and confidence level.

## Model Training

Before using this application, you need to train a ViT model using the `vitTrain.py` script. The training process will save model checkpoints in the `vit_checkpoints` directory, which can then be used by this application.

During training, the script now automatically saves label mappings alongside model checkpoints. These mappings ensure consistency between training and inference.

To train a model:

```bash
python visionTransformer/vitTrain.py
```

### Label Mapping Files

When you train a model, two types of label mapping files are created:

1. **Checkpoint-specific mapping**: Saved alongside each model checkpoint with the naming pattern `[checkpoint_name]_label_map.json`. This file is automatically loaded when you select the corresponding checkpoint in the application.

2. **Timestamped mapping**: Saved in the `label_mappings` directory with a timestamp. These files serve as a backup and reference for all label mappings created during training.

The label mapping files contain:
- `label_to_idx`: Maps text labels to numeric indices
- `idx_to_label`: Maps numeric indices to text labels

## Customization

You can customize the application by:

- Modifying the preprocessing steps in the `preprocess_image` function
- Changing the model architecture or parameters in the `load_model` function
- Adjusting the UI layout and components in the `main` function

## Troubleshooting

If you encounter any issues:

1. Make sure all required packages are installed
2. Check that model checkpoints are available in the `vit_checkpoints` directory
3. Verify that the number of classes specified matches the model's configuration
4. Ensure the uploaded images are in a supported format (JPG, JPEG, PNG)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 