import torch, random, json, torchaudio, numpy as np
device=torch.device("cuda:0"); torch.manual_seed(42); random.seed(42)
from omegaconf import OmegaConf; from ema_pytorch import EMA
cfg=OmegaConf.load("src/YingMusicSinger/config/YingMusic_Singer.yaml")
from src.YingMusicSinger.models.dit import DiT
from src.YingMusicSinger.models.model import Singer
from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
import torch.nn.functional as F

dit=DiT(**cfg.model.arch,text_num_embeds=cfg.datasets_cfg.text_num_embeds,mel_dim=cfg.model.mel_spec.n_mel_channels,long_skip_connection=True)
singer=Singer(transformer=dit,is_tts_pretrain=cfg.model.is_tts_pretrain,melody_input_source=cfg.model.melody_input_source,cka_disabled=cfg.model.cka_disabled,num_channels=None,extra_parameters=cfg.extra_parameters,mel_spec_kwargs=cfg.model.mel_spec,distill_stage=None,use_guidance_scale_embed=False)

# Load BOTH model and ema for comparison
for label, key in [("model", "model_state_dict"), ("ema", "ema_model_state_dict")]:
    ckpt=torch.load("ckpts/plus_ja_sft_v3/step_030000_final.pt",map_location="cpu")
    sd=ckpt[key]
    sd={k.replace("ema_model.","").replace("module.",""):v for k,v in sd.items()}
    singer.load_state_dict(sd,strict=False)
    singer=singer.to(device)
    
    vae=StableAudioInfer(model_config_path="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",model_ckpt_path="ckpts/stable_audio_2_0_vae_20hz_official.ckpt"); vae=vae.to(device).eval()
    midi_teacher=MIDIExtractor(in_dim=80); midi_teacher._load_form_ckpt("ckpts/model_ckpt_steps_100000_simplified.ckpt"); midi_teacher=midi_teacher.to(device).eval()
    mel_spec_ext=MelodySpectrogram(); tokenizer=CNENTokenizer()
    
    with open("${SERVER_ROOT}/final_sum_large/train_singnet.json") as f: data=json.load(f)
    rec=data[0]
    wav,sr=torchaudio.load(rec["Path"])
    if sr!=44100: wav=torchaudio.functional.resample(wav,sr,44100)
    max_s=int(15*44100)
    if wav.shape[-1]>max_s: wav=wav[:,:max_s]
    wav=wav.to(device); text=rec["Text"]
    
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
        midi=singer.smoothMelody_MIDIFuzzDisturb(midi_p)
        midi[:,:ref_len,:]=0
        tokens=tokenizer.encode(text); aligned_text=torch.zeros(B,T,dtype=torch.long,device=device)
        n=min(len(tokens),T); aligned_text[0,:n]=torch.tensor(tokens[:n],device=device)
    
    losses=[]
    for i in range(20):
        t_val=torch.rand(1,device=device)
        noise=torch.randn_like(full_latent)
        x_t=(1-t_val[:,None,None])*noise+t_val[:,None,None]*full_latent
        v_target=full_latent-noise
        
        dit_m=singer.transformer
        time_emb=dit_m.time_embed(t_val)
        x,_=dit_m.get_input_embed(x_t,cond,aligned_text,midi,drop_audio_cond=False,drop_text=False,drop_midi=False,cache=False)
        rope=dit_m.rotary_embed.forward_from_seq_len(T)
        residual=x
        for block in dit_m.transformer_blocks:
            x=block(x,time_emb,mask=None,rope=rope)
        if dit_m.long_skip_connection is not None:
            x=dit_m.long_skip_connection(torch.cat((x,residual),dim=-1))
        x=dit_m.norm_out(x,time_emb)
        v_pred=dit_m.proj_out(x)
        losses.append(float(F.mse_loss(v_pred,v_target)))
    
    print(f"[{label}] flow_loss [0.1: {np.mean(losses[:5]):.4f}] [0.5: {np.mean(losses[5:10]):.4f}] [0.9: {np.mean(losses[10:15]):.4f}] [avg: {np.mean(losses):.4f}]")

















