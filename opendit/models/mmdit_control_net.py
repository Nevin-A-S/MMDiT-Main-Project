import torch
from torch import nn
from einops import rearrange
from .mmdit import MMDiT_models
import torch.distributed as dist
from mmdit import MMDiTBlock as MMDitBlocks
from timm.models.vision_transformer import PatchEmbed
from opendit.embed.pos_emb import  get_2d_sincos_pos_embed
from opendit.utils.operation import gather_forward_split_backward

class MMdit_ControlNet(torch.nn.Module):
    def __init__(self , model : str ,model_config , model_path , fabric):
        super(MMdit_ControlNet, self).__init__()
        device = 'cuda'
        model_config = {
            "input_size": 256 // 8,
            "num_classes": 1000,
            "clip_text_encoder": "openai/clip-vit-base-patch32",
            "t5_text_encoder": "google-t5/t5-small",
        }
        print("Preparing Model")
        self.mmditModel = MMDiT_models[model](**model_config).to(device)
        print("Model Loaded Sucessfully")

        depth = len(self.mmditModel.blocks)

        print("Loading Checkpoint")
        self.load_checkpoint(model_path,fabric)
        print("Checkpoint Loaded Sucessfully")

        for param in self.mmditModel.parameters():
            param.requires_grad = False
        
        print("Pretrained Model Freezed Sucessfully!")

        # "Control Net Params"

        self.controlNetblocks = nn.ModuleList(
            [
                MMDitBlocks(dim_cond = self.mmditModel.hidden_size,
                dim_text = self.mmditModel.hidden_size,
                dim_image = self.mmditModel.hidden_size,
                qk_rmsnorm = True,
                heads = self.mmditModel.num_heads,
                flash_attn = True)
                for _ in range(depth)
            ]
        )
        
        self.controlNet_img_weight = nn.Parameter(torch.tensor(0.0))
        self.controlNet_cc_weight = nn.Parameter(torch.tensor(0.0))

        self.x_embedder = PatchEmbed(self.mmditModel.input_size, self.mmditModel.patch_size, self.mmditModel.in_channels, self.mmditModel.hidden_size, bias=True)
        self.num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.mmditModel.num_patches, self.mmditModel.hidden_size), requires_grad=False)

        "Weight Initialization"
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, t, c, edges=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        c: (N,) texts -> clip embeddings tensor
        edges: (N, C, H, W) tensor of spatial inputs of the canny edge images
        """
        
        cx = edges.to(self.mmditModel.dtype)
        cx = self.x_embedder(cx) + self.pos_embed 

        x = x.to(self.mmditModel.dtype)
        x = self.mmditModel.x_embedder(x) + self.mmditModel.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.mmditModel.t_embedder(t, dtype=x.dtype)  # (N, D)

        c, ts_proj = self.mmditModel.c_embedder(c, self.mmditModel.training)

        t = t + ts_proj

        if self.mmditModel.sequence_parallel_size > 1:
            x = x.chunk(self.mmditModel.sequence_parallel_size, dim=1)[dist.get_rank(self.mmditModel.sequence_parallel_group)]

        for preTrainedBlock, ControlNetBlock in zip(self.mmditModel.blocks, self.controlNetblocks):
            cx, cc = ControlNetBlock(time_cond=t, image_tokens=cx, text_tokens=c)
            x, c = preTrainedBlock(time_cond=t, image_tokens=x, text_tokens=c)  # (N, T, D)
            x = x + cx * self.controlNet_img_weight
            c = c + cc * self.controlNet_cc_weight

        if self.mmditModel.sequence_parallel_size > 1:
            x = gather_forward_split_backward(x, dim=1, process_group=self.mmditModel.sequence_parallel_group)

        c = t
        x = self.mmditModel.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        
        x = self.mmditModel.unpatchify(x)  # (N, out_channels, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def forward_with_cfg(self, x, t, c, cfg_scale, edges=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        
        # Also duplicate the timesteps to match the batch size
        t_combined = torch.cat([t, t], dim=0)
        
        # If edges is provided, duplicate it as well
        if edges is not None:
            edges_half = edges[: len(edges) // 2]
            edges_combined = torch.cat([edges_half, edges_half], dim=0)
        else:
            edges_combined = None
        
        model_out = self.forward(combined, t_combined, c, edges_combined)
        
        # Rest of your code...
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def rearrange_attention_weights(self, flag="load"):
        for block in self.blocks:
            block.attn.rearrange_fused_weight(block.attn.qkv, flag)
        torch.cuda.empty_cache()

    def load_checkpoint(self ,checkpoint_path: str , fabric):
        """Load checkpoint and return the starting epoch and global step"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = fabric.load(checkpoint_path)
        
        self.mmditModel.load_state_dict(checkpoint['model'])

        return 
