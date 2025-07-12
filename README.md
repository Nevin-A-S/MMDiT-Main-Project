MMDiT: Multi-Modal Diffusion Transformer
This repository contains the official implementation of the Multi-Modal Diffusion Transformer (MMDiT). This project is designed for multi-modal generative tasks, leveraging the power of diffusion models and transformers.

📜 Description
MMDiT is a state-of-the-art generative model that combines the strengths of diffusion models for high-quality sample generation and transformers for effective handling of multi-modal data. This implementation provides a flexible and extensible framework for training and evaluating MMDiT on various datasets.

The core of this project's training methodology is built upon a modified version of the OpenDiT repository , tailored to our specific needs.

✨ Features
Multi-Modal Support: Designed to handle various data modalities, such as text, and images.

High-Quality Generation: Leverages diffusion models to produce high-fidelity MRI Images.

Transformer-Based Architecture: Utilizes a transformer backbone for robust and scalable learning.

Distributed Training: Supports distributed training to accelerate the training process on multiple GPUs.


open_dit/: Contains the core diffusion model training implementation, adapted from the OpenDiT repository.

models/: The implementation of the MMDiT model and its components.

train.py: The main script for training the MMDiT model.

sample.py: The main script for generating samples from a trained model.

utils.py: Utility functions used throughout the project.

vit/ : For vit training

Also have added Vit inference code which can be acssed through a streamlit GUI.
