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

dit=DiT(**cfg.model.arch,text_num_embeds=cfg.datasets_cfg.text_num_embeds,mel_dim=cfg.model.mel_spec.n_mel_channels,long_skip_connection=True)
singer=Singer(transformer=dit,is_tts_pretrain=cfg.model.is_tts_pretrain,melody_input_source=cfg.model.melody_input_source,cka_disabled=cfg.model.cka_disabled,num_channels=None,extra_parameters=cfg.extra_parameters,mel_spec_kwargs=cfg.model.mel_spec,distill_stage=None,use_guidance_scale_embed=False)
ckpt=torch.load("ckpts/plus_ja_sft_v3/step_030000_final.pt",map_location="cpu")
sd=ckpt["ema_model_state_dict"]; sd={k.replace("ema_model.",""):v for k,v in sd.items()}
singer.load_state_dict(sd,strict=False); singer=singer.to(device).eval()
vae=StableAudioInfer(model_config_path="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",model_ckpt_path="ckpts/stable_audio_2_0_vae_20hz_official.ckpt"); vae=vae.to(device).eval()
midi_teacher=MIDIExtractor(in_dim=80); midi_teacher._load_form_ckpt("ckpts/model_ckpt_steps_100000_simplified.ckpt"); midi_teacher=midi_teacher.to(device).eval()
mel_spec_ext=MelodySpectrogram(); tokenizer=CNENTokenizer()

with open("${SERVER_ROOT}/final_sum_large/test_singnet.json") as f: data=json.load(f)
rec=data[0]; wav,sr=torchaudio.load(rec["Path"])
if sr!=44100: wav=torchaudio.functional.resample(wav,sr,44100)
wav=wav[:,:int(15*44100)].to(device)
ref_text=rec["Text"]

# Use a different sample's text for "different text"
rec2=data[3] if len(data)>3 else data[1]
diff_text=rec2["Text"]

# Blank text
blank_text_tokens=torch.zeros(1,322,dtype=torch.long,device=device)  # will be resized

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

    def make_text_tokens(text):
        tokens=tokenizer.encode(text)
        tids=torch.zeros(1,T,dtype=torch.long,device=device)
        n=min(len(tokens),T); tids[0,:n]=torch.tensor(tokens[:n],device=device)
        return tids
    
    text_A = make_text_tokens(ref_text)
    text_B = make_text_tokens(diff_text)
    text_blank = torch.zeros(1,T,dtype=torch.long,device=device)

    print(f"Audio T={T}, ref_len={ref_len}")
    print(f"Text A: {ref_text[:40]}")
    print(f"Text B: {diff_text[:40]}")
    print(f"Token A: {text_A[0,:10].tolist()}...")
    print(f"Token B: {text_B[0,:10].tolist()}...")

    # Test at multiple t values and with 3 different random seeds
    noise=torch.randn(1,T,D,device=device)
    x_t=0.5*noise+0.5*full_latent[:,:1,:].expand(-1,T,-1)
    
    for t_val in [0.3, 0.5, 0.7, 0.9]:
        t_tensor=torch.tensor([t_val],device=device)
        v_A,_=singer.transformer(x=x_t,cond=cond,text=text_A,time=t_tensor,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
        v_B,_=singer.transformer(x=x_t,cond=cond,text=text_B,time=t_tensor,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
        v_Z,_=singer.transformer(x=x_t,cond=cond,text=text_blank,time=t_tensor,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
        
        d_AB=float(F.mse_loss(v_A,v_B).sqrt())
        d_AZ=float(F.mse_loss(v_A,v_Z).sqrt())
        d_BZ=float(F.mse_loss(v_B,v_Z).sqrt())
        cos_AB=float(F.cosine_similarity(v_A.flatten(),v_B.flatten(),dim=0))
        cos_AZ=float(F.cosine_similarity(v_A.flatten(),v_Z.flatten(),dim=0))
        
        print(f"\nt={t_val:.1f}:")
        print(f"  L2(A,B)={d_AB:.5f}  L2(A,Z)={d_AZ:.5f}  L2(B,Z)={d_BZ:.5f}")
        print(f"  cos(A,B)={cos_AB:.5f}  cos(A,Z)={cos_AZ:.5f}")
        
        # Check: is v_A-v_B direction stable across seeds?
        diffs_AB=[]
        diffs_AZ=[]
        for seed in range(5):
            torch.manual_seed(seed)
            noise2=torch.randn(1,T,D,device=device)
            x_t2=0.5*noise2+0.5*full_latent[:,:1,:].expand(-1,T,-1)
            v_a2,_=singer.transformer(x=x_t2,cond=cond,text=text_A,time=t_tensor,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
            v_b2,_=singer.transformer(x=x_t2,cond=cond,text=text_B,time=t_tensor,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
            v_z2,_=singer.transformer(x=x_t2,cond=cond,text=text_Z,time=t_tensor,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
            diffs_AB.append(v_a2-v_b2)
            diffs_AZ.append(v_a2-v_z2)
        
        # Cosine between diff vectors across seeds
        cos_ab_cross=[float(F.cosine_similarity(diffs_AB[i].flatten(),diffs_AB[j].flatten(),dim=0)) for i in range(5) for j in range(i+1,5)]
        cos_az_cross=[float(F.cosine_similarity(diffs_AZ[i].flatten(),diffs_AZ[j].flatten(),dim=0)) for i in range(5) for j in range(i+1,5)]
        print(f"  cos(vA-vB across seeds): mean={np.mean(cos_ab_cross):.4f} min={np.min(cos_ab_cross):.4f}")
        print(f"  cos(vA-vZ across seeds): mean={np.mean(cos_az_cross):.4f} min={np.min(cos_az_cross):.4f}")

















