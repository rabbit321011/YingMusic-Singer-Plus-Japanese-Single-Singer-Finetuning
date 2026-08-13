import torch
from omegaconf import OmegaConf
cfg=OmegaConf.load("src/YingMusicSinger/config/YingMusic_Singer.yaml")
from src.YingMusicSinger.models.dit import DiT
from src.YingMusicSinger.models.model import Singer
dit=DiT(**cfg.model.arch,text_num_embeds=cfg.datasets_cfg.text_num_embeds,mel_dim=cfg.model.mel_spec.n_mel_channels,long_skip_connection=True)
singer=Singer(transformer=dit,is_tts_pretrain=cfg.model.is_tts_pretrain,melody_input_source=cfg.model.melody_input_source,cka_disabled=cfg.model.cka_disabled,num_channels=None,extra_parameters=cfg.extra_parameters,mel_spec_kwargs=cfg.model.mel_spec,distill_stage=None,use_guidance_scale_embed=False)
ckpt=torch.load("ckpts/plus_ja_sft_v3/step_030000_final.pt",map_location="cpu")
sd=ckpt.get("model_state_dict",ckpt)
sd={k.replace("module.",""):v for k,v in sd.items()}
singer.load_state_dict(sd,strict=False)
singer=singer.to("cuda:0")
torch.manual_seed(42)
B,T,D=1,322,64; f="cuda:0"
x=torch.randn(B,T,D,device=f)
c=torch.randn(B,T,D,device=f)
txt=torch.zeros(B,T,dtype=torch.long,device=f); txt[0,:10]=100
t=torch.tensor([0.5],device=f)
m=torch.randn(B,T,128,device=f)

singer.eval()
ve,_=singer.transformer(x=x,cond=c,text=txt,time=t,midi=m,drop_audio_cond=False,drop_text=False,drop_midi=False)
singer.train()
vt,_=singer.transformer(x=x,cond=c,text=txt,time=t,midi=m,drop_audio_cond=False,drop_text=False,drop_midi=False)
print(f"eval: mean={ve.mean():.3f} std={ve.std():.3f}")
print(f"train: mean={vt.mean():.3f} std={vt.std():.3f}")
print(f"diff L2={float(torch.nn.functional.mse_loss(ve,vt).sqrt()):.6f}")

















