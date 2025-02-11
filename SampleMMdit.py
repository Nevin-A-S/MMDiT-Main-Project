import argparse
import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from pathlib import Path

from opendit.diffusion import create_diffusion
from opendit.models.mmdit import MMDiT_models
from opendit.utils.download import find_model
from opendit.vae.wrapper import AutoencoderKLWrapper

# Enable TF32 for faster sampling
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path with --ckpt.")

    # Load model:
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Configure input size
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    if args.use_video:
        # Wrap the VAE in a wrapper that handles video data
        vae = AutoencoderKLWrapper(vae)
        input_size = (args.num_frames, args.image_size, args.image_size)
        for i in range(3):
            assert input_size[i] % vae.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // vae.patch_size[i] for i in range(3)]
    else:
        input_size = args.image_size // 8

    # Initialize model
    model_config = {
        "input_size": input_size,
        "clip_text_encoder": args.text_encoder,
        "t5_text_encoder": args.t5_text_encoder,
        "use_video": args.use_video
    }

    model = MMDiT_models[args.model](**model_config).to(device)

    # Load checkpoint
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # Important!

    # Setup diffusion
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Create sampling noise:
    if args.use_video:
        # Sample text prompts
        text_prompts = [
            "A person riding a bicycle through a scenic mountain trail",
            "A beautiful sunset over the ocean",
            "A cat playing with a ball of yarn",
            "A bustling city street at night",
            "A peaceful garden with blooming flowers"
        ]
        n = len(text_prompts)
        z = torch.randn(n, vae.out_channels, *input_size, device=device)
        y = text_prompts * 2  # Duplicate for classifier-free guidance
    else:
        # Sample text prompts
        text_prompts = [
            "a young boy skateboarder jumping on a platform on a skateboard",
            "A white dog is running down a rocky beach",
            "A boy smiles for the camera at a beach",
            "a Man",
            "a girl",
            "Flower",
            "car",
            "hairpin",
            "A little kid playing GameCube at McDonald 's",
            "The reunion is in full swing with a moon bounce .",
            "Two people are reading on a bench .",
            " Runner being cheered on by the crowd .",
            " A brown dog chases the water from a sprinkler on a lawn .",
             ' A person in the snow drilling a hole in the ice .',
             ' A guy in shorts and a white t-shirt sits on the ground in front of a grill with hotdogs on it .',
             ' Man sweeping the street outside .',
             ' A man with a black shirt giving another man a tattoo .',
             ' A barefooted man wearing olive green shorts grilling hotdogs on a small propane grill while holding a blue plastic cup .',
             ' Two men hiking in the snow .', ' A bearded traveler in a red shirt sitting in a car and reading a map .',
             ' A young adult wearing rollerblades , holding a cellular phone to her ear .',
             ' Four people walking through the sunset in clear blue skies .',
             ' A young man in sunglasses holds a blue cup next to a grill with sausages .',
             ' A man in a blue shirt driving a Segway type vehicle .', 
             ' Two large tan dogs play along a sandy beach .',
             ' A man getting a tattoo on his back .',
             ' A man with gauges and glasses is wearing a Blitz hat .',
             ' Two men on a rooftop while another man stands atop a ladder watching them',
             ' A woman in a headscarf uses a telescope to look out over the city and bay .',
             ' a boy in a black t-shirt and blue jeans is pushing a toy three wheeler around a small pool .',
             ' Man in black coat examining airplane nose .',
             ' Young girl with pigtails painting outside in the grass .',
             ' Several climbers in a row are climbing the rock while the man in red watches and holds the line .',
             ' A man in a blue hard hat and orange safety vest stands in an intersection while holding a flag .',
             ' A man in a blue shirt is standing on a ladder cleaning a window .',
             ' Five ballet dancers caught mid jump in a dancing studio with sunlight coming through a window .',
             ' Climber climbing an ice wall',
             " A man sitting with his daughter in his daughter 's pink car in a grocery store .",
             ' 2 female babies eating chips .',
             ' The black dog runs through the water .',
             ' A woman is sorting white tall candles as a man in a green shirt stands behind her .',
             ' Two friends enjoy time spent together .',
             ' A young man in a black and yellow jacket is gazing at something and smiling .',
             ' a man on a ladder cleans a window'

        ]
        n = len(text_prompts)
        z = torch.randn(n, 4, input_size, input_size, device=device)
        y = text_prompts * 2  # Duplicate for classifier-free guidance

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    model_kwargs = dict(c=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # Save and display images:
    if args.use_video:
        samples = vae.decode(samples)
        # Save video samples
        save_path = Path(args.output_dir) / "samples_video"
        save_path.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            video_path = save_path / f"sample_{i}.mp4"
            # Save video using appropriate method (you may need to implement this)
            print(f"Saved video sample to {video_path}")
    else:
        samples = vae.decode(samples / 0.18215).sample
        # Save image samples
        save_path = Path(args.output_dir) / f"samples_image_{ckpt_path}"
        save_path.mkdir(parents=True, exist_ok=True)
        for i, (sample, prompt) in enumerate(zip(samples, text_prompts)):
            image_path = save_path / f"sample_{i}.png"
            save_image(sample, image_path, normalize=True, value_range=(-1, 1))
            # Save prompt
            prompt_path = save_path / f"sample_{i}_prompt.txt"
            prompt_path.write_text(prompt)
            print(f"Saved image sample to {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MMDiT_models.keys()), default="MMDiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--use_video", action="store_true", help="Use video data instead of images")
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--t5_text_encoder", type=str, default="google-t5/t5-small")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a MMDiT checkpoint",
    )
    args = parser.parse_args()
    main(args)
