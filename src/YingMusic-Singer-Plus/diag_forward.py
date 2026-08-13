import torch, json, torch.nn.functional as F, torchaudio, numpy as np
device=torch.device("cuda:0"); torch.manual_seed(42)
from omegaconf import OmegaConf
cfg=OmegaConf.load("src/YingMusicSinger/config/YingMusic_Singer.yaml")
from src.YingMusicSinger.models.dit import DiT
from src.YingMusicSinger.models.model import Singer
from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
from ema_pytorch import EMA

dit=DiT(**cfg.model.arch,text_num_embeds=cfg.datasets_cfg.text_num_embeds,mel_dim=cfg.model.mel_spec.n_mel_channels,long_skip_connection=True)
singer=Singer(transformer=dit,is_tts_pretrain=cfg.model.is_tts_pretrain,melody_input_source=cfg.model.melody_input_source,cka_disabled=cfg.model.cka_disabled,num_channels=None,extra_parameters=cfg.extra_parameters,mel_spec_kwargs=cfg.model.mel_spec,distill_stage=None,use_guidance_scale_embed=False)
ckpt=torch.load("ckpts/plus_ja_sft_v3/step_030000_final.pt",map_location="cpu")
sd=ckpt.get("ema_model_state_dict",ckpt.get("model_state_dict",ckpt))
sd={k.replace("ema_model.","").replace("module.",""):v for k,v in sd.items()}
singer.load_state_dict(sd,strict=False); singer=singer.to(device).eval()
vae=StableAudioInfer(model_config_path="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",model_ckpt_path="ckpts/stable_audio_2_0_vae_20hz_official.ckpt"); vae=vae.to(device).eval()
midi_teacher=MIDIExtractor(in_dim=80); midi_teacher._load_form_ckpt("ckpts/model_ckpt_steps_100000_simplified.ckpt"); midi_teacher=midi_teacher.to(device).eval()
mel_spec_ext=MelodySpectrogram(); tokenizer=CNENTokenizer()

with open("${SERVER_ROOT}/final_sum_large/test_singnet.json") as f: data=json.load(f)
rec=data[0]
wav,sr=torchaudio.load(rec["Path"])
if sr!=44100: wav=torchaudio.functional.resample(wav,sr,44100)
max_s=int(15*44100)
if wav.shape[-1]>max_s: wav=wav[:,:max_s]
wav=wav.to(device); ref_text=rec["Text"]

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
    tokens=tokenizer.encode(ref_text)
    text_ids=torch.zeros(1,T,dtype=torch.long,device=device)
    n=min(len(tokens),T); text_ids[0,:n]=torch.tensor(tokens[:n],device=device)

    # === KEY TEST: mimic training forward exactly, then mimic inference ===
    noise=torch.randn(1,T,D,device=device)
    
    # Training-style: x_t = (1-t)*noise + t*full_latent, t=0.5
    x_train = 0.5*noise + 0.5*full_latent
    v_target = full_latent - noise
    
    t_tensor = torch.tensor([0.5],device=device)
    v_pred_train,_ = singer.transformer(x=x_train, cond=cond, text=text_ids, time=t_tensor, midi=midi, drop_audio_cond=False, drop_text=False, drop_midi=False)
    
    loss_train = float(F.mse_loss(v_pred_train, v_target))
    print(f"[Training-like forward at t=0.5]")
    print(f"  v_pred mean={v_pred_train.mean():.3f} std={v_pred_train.std():.3f}")
    print(f"  v_target mean={v_target.mean():.3f} std={v_target.std():.3f}")
    print(f"  MSE loss = {loss_train:.4f}")
    print(f"  Cosine sim(v_pred, v_target) = {float(F.cosine_similarity(v_pred_train.flatten(), v_target.flatten(), dim=0)):.4f}")
    
    # Infer-style x_t
    x_infer = 0.5*noise + 0.5*full_latent[:,:1,:].expand(-1,T,-1)
    v_pred_infer,_ = singer.transformer(x=x_infer, cond=cond, text=text_ids, time=t_tensor, midi=midi, drop_audio_cond=False, drop_text=False, drop_midi=False)
    v_target_infer = full_latent - noise
    loss_infer = float(F.mse_loss(v_pred_infer, v_target_infer))
    
    print(f"\n[Inference-like forward at t=0.5]")
    print(f"  v_pred mean={v_pred_infer.mean():.3f} std={v_pred_infer.std():.3f}")
    print(f"  MSE loss = {loss_infer:.4f} (same v_target)")
    print(f"  L2(v_pred_train, v_pred_infer) = {float(F.mse_loss(v_pred_train, v_pred_infer).sqrt()):.4f}")
    
    # Generate with training-style init (full latent as oracle)
    print(f"\n=== Generation: different initializations ===")
    for label, x_init in [("infer-style", x_infer), ("train-style", x_train)]:
        nfe=32; t_vals=torch.linspace(0.5,1,nfe+1,device=device); t_vals=0.5*t_vals/(1+(0.5-1)*t_vals)
        x=x_init
        for i in range(len(t_vals)-1):
            dt_val=t_vals[i+1]-t_vals[i]; t_val=t_vals[i]
            v_cond,_=singer.transformer(x=x,cond=cond,text=text_ids,time=t_val,midi=midi,drop_audio_cond=False,drop_text=False,drop_midi=False)
            x=x+v_cond*dt_val.item()
        lat_dec=x.permute(0,2,1).float()
        audio_gen=vae.decode_audio(lat_dec)
        mse_to_real=float(F.mse_loss(x, full_latent))
        print(f"  {label:15s}: latent std={x.std():.3f} MSE_to_real={mse_to_real:.4f} audio std={audio_gen.std():.3f}")
    
    # Also test direct reconstruction: x = cond (first 50% is real, what about rest?)
    print(f"\n=== Direct: what if we just decode the real latent? ===")
    lat_dec_real=full_latent.permute(0,2,1).float()
    audio_real=vae.decode_audio(lat_dec_real)
    rn=audio_real.squeeze().cpu().numpy()
    if rn.ndim>1: rn=rn.T
    print(f"  Real latent decode: audio mean={audio_real.mean():.3f} std={audio_real.std():.3f}")
    
    # What if we replace the generated region with real latent?
    # (to check if VAE decode is working correctly for generator region)
    print(f"\n=== Hybrid: cond+real gen region ===")
    x_hybrid = full_latent.clone()
    x_hybrid[:,ref_len:,:] = torch.randn(1,T-ref_len,D,device=device)*0.5  # replace gen region with noise
    lat_dec_h=x_hybrid.permute(0,2,1).float()
    audio_hybrid=vae.decode_audio(lat_dec_h)
    print(f"  cond(real)+gen(noise) decode: audio std={audio_hybrid.std():.3f}")

















