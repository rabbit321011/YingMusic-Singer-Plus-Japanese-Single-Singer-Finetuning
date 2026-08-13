import torch, json, torchaudio, numpy as np, torch.nn.functional as F
device="cuda:0"; torch.manual_seed(42)
from omegaconf import OmegaConf
cfg=OmegaConf.load("src/YingMusicSinger/config/YingMusic_Singer.yaml")
from src.YingMusicSinger.models.dit import DiT
from src.YingMusicSinger.models.model import Singer
from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer

vae=StableAudioInfer(model_config_path="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",model_ckpt_path="ckpts/stable_audio_2_0_vae_20hz_official.ckpt"); vae=vae.to(device).eval()
midi_teacher=MIDIExtractor(in_dim=80); midi_teacher._load_form_ckpt("ckpts/model_ckpt_steps_100000_simplified.ckpt"); midi_teacher=midi_teacher.to(device).eval()
mel_spec_ext=MelodySpectrogram(); tokenizer=CNENTokenizer()

with open("${SERVER_ROOT}/final_sum_large/test_singnet.json") as f: data=json.load(f)
rec=data[0]; wav,sr=torchaudio.load(rec["Path"])
if sr!=44100: wav=torchaudio.functional.resample(wav,sr,44100)
wav=wav[:,:int(15*44100)].to(device)

with torch.no_grad():
    w2d=wav.squeeze(0) if wav.dim()==2 else wav
    if w2d.dim()==1: w2d=w2d.unsqueeze(0)
    lat=vae.encode_audio(w2d,in_sr=44100)
    if lat.dim()==3: lat=lat.squeeze(0)
    full_latent=lat.transpose(0,1).unsqueeze(0); B,T,D=full_latent.shape
    ref_len=T//2; cond=torch.zeros_like(full_latent); cond[:,:ref_len,:]=full_latent[:,:ref_len,:]
    mel=mel_spec_ext(audio=w2d,sr=44100)
    if mel.dim()==3: mel=mel.squeeze(0)
    mel=mel.to(device); midi_p,_=midi_teacher(mel.unsqueeze(0).transpose(1,2))
    if midi_p.shape[1]!=T: midi_p=F.interpolate(midi_p.transpose(1,2),size=T,mode="linear",align_corners=False).transpose(1,2)
    midi=midi_p.clone(); midi[:,:ref_len,:]=0
    text_A=tokenizer.encode(rec["Text"])
    text_B=tokenizer.encode(data[3]["Text"])

def load_ckpt(path):
    dit=DiT(**cfg.model.arch,text_num_embeds=cfg.datasets_cfg.text_num_embeds,mel_dim=cfg.model.mel_spec.n_mel_channels,long_skip_connection=True)
    singer=Singer(transformer=dit,is_tts_pretrain=cfg.model.is_tts_pretrain,melody_input_source=cfg.model.melody_input_source,cka_disabled=cfg.model.cka_disabled,num_channels=None,extra_parameters=cfg.extra_parameters,mel_spec_kwargs=cfg.model.mel_spec,distill_stage=None,use_guidance_scale_embed=False)
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    sd=ckpt.get("ema_model_state_dict",ckpt.get("model_state_dict",ckpt))
    sd={k.replace("ema_model.","").replace("module.",""):v for k,v in sd.items()}
    singer.load_state_dict(sd,strict=False)
    return singer.to(device).eval()

def make_tokens(tokens_list):
    tids=torch.zeros(1,T,dtype=torch.long,device=device)
    n=min(len(tokens_list),T); tids[0,:n]=torch.tensor(tokens_list[:n],device=device)
    return tids

tA=make_tokens(text_A); tB=make_tokens(text_B); tZ=torch.zeros(1,T,dtype=torch.long,device=device)
t_tensor=torch.tensor([0.5],device=device)

ckpts=[
    ("base", "ckpts/YingMusicSinger_model.pt"),
    ("step_010000", "ckpts/plus_ja_sft_v3/step_010000.pt"),
    ("step_020000", "ckpts/plus_ja_sft_v3/step_020000.pt"),
    ("step_030000", "ckpts/plus_ja_sft_v3/step_030000_final.pt"),
]

print(f"{'ckpt':>14s}  L2(A,B)  L2(A,Z)  cos(A,B)  cos(A,Z)  v_mean  v_std  cos_crossseed")
print("-"*85)

for label, path in ckpts:
    singer=load_ckpt(path)
    
    # 5 seeds for cross-seed stability
    diffs_AB=[]
    for seed in range(5):
        torch.manual_seed(seed)
        noise=torch.randn(1,T,D,device=device)
        x_t=0.5*noise+0.5*full_latent[:,:1,:].expand(-1,T,-1)
        v_a,_=singer.transformer(x=x_t,cond=cond,text=tA,time=t_tensor,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
        v_b,_=singer.transformer(x=x_t,cond=cond,text=tB,time=t_tensor,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
        v_z,_=singer.transformer(x=x_t,cond=cond,text=tZ,time=t_tensor,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
        if seed==0:
            l2_ab=float(F.mse_loss(v_a,v_b).sqrt())
            l2_az=float(F.mse_loss(v_a,v_z).sqrt())
            cos_ab=float(F.cosine_similarity(v_a.flatten(),v_b.flatten(),dim=0))
            cos_az=float(F.cosine_similarity(v_a.flatten(),v_z.flatten(),dim=0))
            vm=v_a.mean().item(); vs=v_a.std().item()
        diffs_AB.append(v_a-v_b)
    
    cos_cross=[float(F.cosine_similarity(diffs_AB[i].flatten(),diffs_AB[j].flatten(),dim=0)) for i in range(5) for j in range(i+1,5)]
    
    print(f"{label:>14s}  {l2_ab:.5f}  {l2_az:.5f}  {cos_ab:.5f}  {cos_az:.5f}  {vm:6.3f}  {vs:5.3f}  {np.mean(cos_cross):.3f}")
    del singer; torch.cuda.empty_cache()

















