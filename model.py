import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class ActionEmbedder(FeedForward):
    def __init__(
        self,
        action_dim: int = 3,
        embed_dim: int = 1024,
        num_tokens: int = 4,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        act_fn: str = "swiglu",
    ):

        super().__init__(
            dim=action_dim,
            inner_dim=hidden_dim,
            dim_out=embed_dim * num_tokens,
            activation_fn=act_fn,
            dropout=dropout,
        )

        self.num_tokens = num_tokens

    def forward(self, a: torch.Tensor):
        # a: (B, action_dim)
        x = super().forward(a)  # (B, embed_dim * tokens)
        x = x.view(a.shape[0], self.num_tokens, -1)
        return x  # (B, tokens, embed_dim)


class WorldModel(nn.Module):
    def __init__(
        self,
        base_name: str = "runwayml/stable-diffusion-v1-5",
        action_dim: int = 3,
        num_tokens: int = 4,
        hidden_dim: int = 1024,
    ):

        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(base_name, subfolder="unet")

        self.action_embedder = ActionEmbedder(
            action_dim=action_dim,
            embed_dim=self.unet.config.cross_attention_dim,  # pyright: ignore[reportAttributeAccessIssue]
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
        )

        scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        self.num_timesteps = scheduler.config.num_train_timesteps  # pyright: ignore[reportAttributeAccessIssue]

    def forward(self, a_t: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor | None = None):
        cond = self.action_embedder(a_t)  # (B, tokens, dim)
        B = x_t.shape[0]
        if t is None:
            if self.training:
                t = torch.randint(0, self.num_timesteps, (B,), device=x_t.device)
            else:
                t = torch.zeros((B,), dtype=torch.long, device=x_t.device)
        
        pred_delta  = self.unet(
            sample=x_t, timestep=t, encoder_hidden_states=cond
        ).sample
        return pred_delta 
