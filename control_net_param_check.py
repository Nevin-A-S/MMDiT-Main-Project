import torch
from lightning.fabric import Fabric
from opendit.models.mmdit_control_net import MMdit_ControlNet

def load_checkpoint(checkpoint_path: str, model, fabric):
    """Load checkpoint and return the starting epoch and global step"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = fabric.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return checkpoint['epoch'], checkpoint['global_step']

def main():
    # Initialize Fabric
    fabric = Fabric(
        accelerator="cuda",
        devices=1,
        precision="bf16-mixed",
    )
    fabric.launch()

    # Model configuration matching training script
    model_config = {
        "input_size": 256 // 8,
        "num_classes": 1000,
        "clip_text_encoder": "openai/clip-vit-base-patch32",
        "t5_text_encoder": "google-t5/t5-small",
    }

    # Create model instance
    model = MMdit_ControlNet(
        model="MMDiT-B/2",
        model_config=model_config,
        model_path="outputs/009-MMDiT-B-2/checkpoint_1100000.pt",
        fabric=fabric
    )

    # Load the checkpoint from training
    checkpoint_path = "outputs/028-MMDiT-B-2/checkpoint_control_net_0180000.pt"
    epoch, global_step = load_checkpoint(checkpoint_path, model, fabric)

    print(f"\nLoaded checkpoint from epoch {epoch}, global step {global_step}")
    print("\nControlNet Parameters:")
    print(f"controlNet_img_weight: {model.controlNet_img_weight.item():.6f}")
    print(f"controlNet_cc_weight: {model.controlNet_cc_weight.item():.6f}")

if __name__ == "__main__":
    main()
