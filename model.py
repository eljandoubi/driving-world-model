import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


class ActionEmbedder(FeedForward):
    def __init__(self, action_dim:int=3, embed_dim:int=1024,
                 num_tokens:int=4, hidden_dim:int=1024,
                 dropout:float=0.1, act_fn:str="swiglu"):
        
        super().__init__(dim=action_dim,  inner_dim=hidden_dim,
                         dim_out=embed_dim * num_tokens,
                         activation_fn=act_fn, dropout=dropout)

        self.num_tokens = num_tokens

    def forward(self, a:torch.Tensor):
        # a: (B, action_dim)
        x = super().forward(a)  # (B, embed_dim * tokens)
        x = x.view(a.shape[0], self.num_tokens, -1)
        return x  # (B, tokens, embed_dim)
    

class WorldModel(nn.Module):
    def __init__(self, base_name:str="runwayml/stable-diffusion-v1-5",
                 action_dim:int=3, num_tokens:int=4, hidden_dim:int=1024):
        
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
                base_name,
                subfolder="unet"
            )
        
        self.action_embedder = ActionEmbedder(action_dim=action_dim,
                                   embed_dim = self.unet.config.cross_attention_dim, # pyright: ignore[reportAttributeAccessIssue]
                                   num_tokens=num_tokens,
                                   hidden_dim=hidden_dim,
                                   )
        
    def forward(self, a_t:torch.Tensor, x_t: torch.Tensor):
        cond = self.action_embedder(a_t)  # (B, tokens, dim)
        B = x_t.shape[0]
        t = torch.ones((B,),dtype=torch.long,device=x_t.device)
        x_tp1_pred = self.unet(
            sample=x_t,
            timestep=t,
            encoder_hidden_states=cond
        ).sample
        return x_tp1_pred
    
if __name__ == "__main__":
    from torchvision import transforms
    from tqdm import tqdm

    from data_prep import frames
    model = WorldModel().cuda()
    im_s = model.unet.config.sample_size * 4
    print(im_s)
    in_c = model.unet.config.in_channels
    transform = transforms.Compose([
        transforms.Resize((im_s, im_s)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*in_c, std=[0.5]*in_c)
    ])
    x_t = transform(frames["frame_t"]).unsqueeze(0).cuda()
    x_tp1 = transform(frames["frame_tp1"]).unsqueeze(0).cuda()
    a_t = torch.tensor(frames["action"]).unsqueeze(0).cuda()
    for _ in tqdm(range(1)):
        with torch.inference_mode():
            x_tp1_pred = model(a_t,x_t)
            loss = nn.functional.mse_loss(x_tp1_pred,x_tp1)
            print(loss)

