import torch
import torch.nn as nn
import open_clip
from transformers import (
    CLIPModel, 
    CLIPProcessor, 
    T5EncoderModel, 
    T5Tokenizer
)
from typing import Optional, Tuple
import gc

class CaptionEncoder(nn.Module):
    '''
    Forward method of the caption encoder outputs the encoded text using CLIP-G/14, CLIP-L/14, and T5.
    The output is a concatenated tensor of CLIP-G/14, CLIP-L/14, and T5 features. The Output Dimension is (batch_size, 2560) 
    '''
    def __init__(
        self,
        clip_l_name = "openai/clip-vit-large-patch14",
        clip_g_name = "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        t5_name = "google-t5/t5-base",
        device = "cuda" if torch.cuda.is_available() else "cpu",
        max_length =77,
        offloading = True
    ):
        super().__init__()
        self.device = device
        
        self.clip_l_name = clip_l_name
        self.t5_name = t5_name
        self.clip_g_name = clip_g_name
        
        self.clip_g = None
        self.clip_g_tokenizer = None
        self.clip_l = None
        self.clip_l_processor = None
        self.t5 = None
        self.t5_tokenizer = None
        self.max_length = max_length
        self.offloading = offloading

    ###################- Model Loading and Unloading Helper functions -##################
    
    def _load_clip_g(self):
        """Load CLIP-G/14 model and tokenizer."""
        if self.clip_g is None:
            self.clip_g, _, _ = open_clip.create_model_and_transforms(
                self.clip_g_name,
                cache_dir= './PreTrainedModels/clip-g-14'
            )
            self.clip_g = self.clip_g.to(self.device)
            self.clip_g_tokenizer = open_clip.get_tokenizer(
                self.clip_g_name
            )
            for param in self.clip_g.parameters():
                param.required_grad=False
    
    def _unload_clip_g(self):
        """Unload CLIP-G/14 model and tokenizer."""
        if self.clip_g is not None and self.offloading == True:
            self.clip_g = self.clip_g.to('cpu')
            del self.clip_g
            del self.clip_g_tokenizer
            self.clip_g = None
            self.clip_g_tokenizer = None
            torch.cuda.empty_cache()
            gc.collect()
    
    def _load_clip_l(self):
        """Load CLIP-L/14 model and processor."""
        if self.clip_l is None :
            self.clip_l = CLIPModel.from_pretrained(self.clip_l_name, cache_dir= './PreTrainedModels/clip-l-14').to(self.device)
            self.clip_l_processor = CLIPProcessor.from_pretrained(self.clip_l_name , cache_dir= './PreTrainedModels/clip-l-14')
            for param in self.clip_l.parameters():
                param.required_grad=False
    
    def _unload_clip_l(self):
        """Unload CLIP-L/14 model and processor."""
        if self.clip_l is not None and self.offloading == True:
            self.clip_l = self.clip_l.to('cpu')
            del self.clip_l
            del self.clip_l_processor
            self.clip_l = None
            self.clip_l_processor = None
            torch.cuda.empty_cache()
            gc.collect()
    
    def _load_t5(self):
        """Load T5 model and tokenizer."""
        if self.t5 is None:
            self.t5 = T5EncoderModel.from_pretrained(self.t5_name , cache_dir= './PreTrainedModels/t5').to(self.device)
            self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_name , cache_dir= './PreTrainedModels/t5')
            for param in self.t5.parameters():
                param.required_grad=False
    
    def _unload_t5(self):
        """Unload T5 model and tokenizer."""
        if self.t5 is not None and self.offloading == True:
            self.t5 = self.t5.to('cpu')
            del self.t5
            del self.t5_tokenizer
            self.t5 = None
            self.t5_tokenizer = None
            torch.cuda.empty_cache()
            gc.collect()

    ###################- Model Process Logic -################################

    def _process_clip_g(self, captions: list[str]) -> torch.Tensor:
        """Process captions through CLIP-G/14 model."""
        self._load_clip_g()
        tokens = self.clip_g_tokenizer(captions).to(self.device)

        token_embeddings = self.clip_g.token_embedding(tokens)  

        # print(token_embeddings.shape)  

        outputs = self.clip_g.transformer(token_embeddings)
    
        z = outputs.detach().clone()  
        self._unload_clip_g()
        pooled_z = z.mean(dim=1)  
    
        return z , pooled_z
    
    def _process_clip_l(self, captions: list[str]) -> torch.Tensor:
        """Process captions through CLIP-L/14 model."""
        self._load_clip_l()
        batch_encoding = self.clip_l_processor(
            captions,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.clip_l.text_model(input_ids=tokens)
        # print(type(outputs))
    
        z = outputs.last_hidden_state
        pooled_z = outputs.pooler_output
    
        z = z.detach().clone()  # Create a copy before unloading
        self._unload_clip_l()
        return z , pooled_z
    
    def _process_t5(self, captions: list[str]) -> torch.Tensor:
        """Process captions through T5 model."""
        self._load_t5()
        
        batch_encoding = self.t5_tokenizer(
            captions,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.t5(input_ids=tokens)

        z = outputs.last_hidden_state
        # pooled_z = outputs.pooler_output

        return z 


    @torch.no_grad()
    def forward(self, captions):
        """
        Forward pass through the caption encoder.
        Models are loaded and unloaded sequentially to save memory.
        
        Args:
            captions: List of caption strings
            
        Returns:
            Tensor of shape (batch_size, 2560)
        """
        clip_l_features , clip_l_features_pooled = self._process_clip_l(captions)

        clip_g_features ,clip_g_features_pooled = self._process_clip_g(captions)

        t5_features = self._process_t5(captions)

        combined_features = torch.cat(
            [clip_l_features, clip_g_features],
            dim=-1
        )
        # print('shape clip comb')
        # print(combined_features.shape)

        t5_padded = torch.nn.functional.pad(t5_features , ( 0,1792-768, 0 ,0  ))
        
        # print('shape t5 paded')
        # print(t5_padded.shape)

        combined_features = torch.cat([combined_features , t5_padded], dim =1)

        # print('Out comb')
        # print(combined_features.shape)

        pooled_features = torch.cat([clip_l_features_pooled,clip_g_features_pooled] , dim=1)

        # print('pooled comb')
        # print(pooled_features.shape)
        return combined_features , pooled_features

if __name__ == "__main__":
    import time

    start_time = time.time()
    
    model = CaptionEncoder(
        clip_l_name="openai/clip-vit-large-patch14",
        clip_g_name="hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        t5_name="google-t5/t5-base",
        offloading=False)
    
    captions = [
        "A beautiful sunset over the mountains A beautiful sunset over the mountains A beautiful sunset over the mountains A beautiful sunset over the mountains",
        'anime car'
    ]

    output = model(captions)
    print(output)
    print(f"Output shape: {output[0].shape}") 
    
    captions = [
        "A beautiful sunset over the mountains A beautiful sunset over the mountains",
        'anime car'
    ]

    output = model(captions)

    output = model(captions)

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", execution_time, "seconds")