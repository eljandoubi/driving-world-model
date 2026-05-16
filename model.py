import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


class ActionEmbedder(nn.Module):
    def __init__(self, action_dim:int=3, embed_dim:int=128,
                 num_tokens:int=4, hidden_dim:int=128):
        super().__init__()

        self.num_tokens = num_tokens

        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * num_tokens)
        )

    def forward(self, a:torch.Tensor):
        # a: (B, action_dim)
        x = self.mlp(a)  # (B, embed_dim * tokens)
        x = x.view(a.shape[0], self.num_tokens, -1)
        return x  # (B, tokens, embed_dim)
    

class WorldModel(nn.Module):
    def __init__(self, base_name:str="google/ddpm-cifar10-32",
                 action_dim:int=3, num_tokens:int=4, hidden_dim:int=128):
        
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            "google/ddpm-cifar10-32"
        )
        self.action_embedder = ActionEmbedder(action_dim=action_dim,
                                   embed_dim = self.unet.config.cross_attention_dim, # pyright: ignore[reportAttributeAccessIssue]
                                   num_tokens=num_tokens,
                                   hidden_dim=hidden_dim,
                                   )
        
    def forward(self, a_t:torch.Tensor,noisy_x: torch.Tensor, t: torch.Tensor):
        # 🔥 action embedding
        cond = self.action_embedder(a_t)  # (B, tokens, dim)

        # forward
        noise_pred = self.unet(
            sample=noisy_x,
            timestep=t,
            encoder_hidden_states=cond
        ).sample
        return noise_pred