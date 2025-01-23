import torch
from mmdit import MMDiTBlocks

# define mm dit block

block = MMDiTBlock(
    dim_cond = 256,
    dim_text = 768,
    dim_image = 512,
    qk_rmsnorm = True
)

# mock inputs

time_cond = torch.randn(2, 256)

text_tokens = torch.randn(2, 512, 768)
text_mask = torch.ones((2, 512)).bool()

image_tokens = torch.randn(2, 1024, 512)

# single block forward

text_tokens_next, image_tokens_next = block(
    time_cond = time_cond,
    text_tokens = text_tokens,
    text_mask = text_mask,
    image_tokens = image_tokens
)

print(text_tokens_next.shape) 
print(image_tokens_next.shape) 

# multiple blocks forward

blocks = [MMDiTBlocks(dim_cond = 256, dim_text = 768, dim_image = 512, qk_rmsnorm = True) for _ in range(3)]