import json, os, re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import accelerate

CKPT = '${REMOTE_ROOT}/ckpts/Vevo2/contentstyle_modeling/posttrained'
DATA_DIR = '${REMOTE_ROOT}/final_sum_large'
OUTDIR = '${REMOTE_ROOT}/ckpts/ar_target singer_v3'
EPOCHS = 25
LR = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LEN = 2048
SAVE_EPOCHS = 2
LOG_STEPS = 50
WARMUP_RATIO = 0.04

os.makedirs(OUTDIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open(os.path.join(DATA_DIR, 'train_singnet.json'), 'r') as f:
    train_raw = json.load(f)
with open(os.path.join(DATA_DIR, 'test_singnet.json'), 'r') as f:
    test_raw = json.load(f)

print(f'Train records: {len(train_raw)} | Test records: {len(test_raw)}')

codes_dir = os.path.join(DATA_DIR, 'codes')

def gen_sample(rec):
    wav_path = rec['Path']
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    code_path = os.path.join(codes_dir, f'{stem}.pt')
    if not os.path.exists(code_path):
        return None
    codes = torch.load(code_path, map_location='cpu', weights_only=True)
    cs_ids = codes['cs_ids'].tolist()
    prosody_ids = codes['prosody_ids'].tolist()
    return rec['Text'], cs_ids, prosody_ids

train_samples = []
for rec in tqdm(train_raw, desc='Loading train'):
    s = gen_sample(rec)
    if s:
        train_samples.append(s)
test_samples = []
for rec in tqdm(test_raw, desc='Loading test'):
    s = gen_sample(rec)
    if s:
        test_samples.append(s)

print(f'Train samples: {len(train_samples)} | Test samples: {len(test_samples)}')

# Compute step counts
n_gpu = 6
effective_batch = BATCH_SIZE * GRAD_ACCUM * n_gpu
steps_per_epoch = (len(train_samples) + effective_batch - 1) // effective_batch
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
save_steps = steps_per_epoch * SAVE_EPOCHS

print(f'Effective batch: {effective_batch}')
print(f'Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Warmup: {warmup_steps}')
print(f'Save every: {SAVE_EPOCHS} epochs = {save_steps} steps')
print(f'ETA: ~{total_steps / 1.7 / 3600:.1f}h at 1.7 it/s')

SYSTEM_PROMPT = 'User will provide you with a text. Please vocalize it with natural expression.'

def build_training_text(text, prosody_ids, cs_ids):
    prosody_str = ''.join([f'<|prosody_{int(i)}|>' for i in prosody_ids])
    prosody_str = '<|prosody_start|>' + prosody_str + '<|prosody_end|>'
    cs_str = ''.join([f'<|content_style_{int(i)}|>' for i in cs_ids])
    cs_str = '<|content_style_start|>' + cs_str + '<|content_style_end|>'
    return (
        f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
        f'<|im_start|>user\n{text}<|im_end|>\n'
        f'<|im_start|>assistant\n{prosody_str}{cs_str}<|im_end|>'
    )

def tokenize(samples):
    ids_list, labs_list = [], []
    for text, cs_ids, prosody_ids in tqdm(samples, desc='Tokenizing'):
        full = build_training_text(text, prosody_ids, cs_ids)
        ids = tokenizer.encode(full, add_special_tokens=False)
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        labels = [-100] * len(ids)
        asst = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
        for i in range(len(ids) - len(asst) + 1):
            if ids[i:i+len(asst)] == asst:
                for j in range(i + len(asst), len(ids)):
                    labels[j] = ids[j]
                break
        ids_list.append(torch.tensor(ids, dtype=torch.long))
        labs_list.append(torch.tensor(labels, dtype=torch.long))
    return ids_list, labs_list

class ArDataset(Dataset):
    def __init__(self, ids_list, labs_list):
        self.ids = ids_list
        self.labs = labs_list
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        return {
            'input_ids': self.ids[idx],
            'labels': self.labs[idx],
            'attention_mask': torch.ones(len(self.ids[idx]), dtype=torch.long),
        }

def collate_fn(batch):
    ml = max(x['input_ids'].size(0) for x in batch)
    pi, pl, pa = [], [], []
    for x in batch:
        pad = ml - x['input_ids'].size(0)
        pi.append(torch.cat([x['input_ids'], torch.full((pad,), tokenizer.pad_token_id, dtype=torch.long)]))
        pl.append(torch.cat([x['labels'], torch.full((pad,), -100, dtype=torch.long)]))
        pa.append(torch.cat([x['attention_mask'], torch.zeros(pad, dtype=torch.long)]))
    return {'input_ids': torch.stack(pi), 'labels': torch.stack(pl), 'attention_mask': torch.stack(pa)}

train_ids, train_labs = tokenize(train_samples)
test_ids, test_labs = tokenize(test_samples)
train_dataset = ArDataset(train_ids, train_labs)
eval_dataset = ArDataset(test_ids, test_labs)

model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16)
model.train()

class MyTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        dl = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        all_losses = []
        for batch in dl:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = batch['labels'][..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            mask = labels.reshape(-1) != -100
            if mask.any():
                all_losses.append(loss[mask].mean().item())
        max_loss = max(all_losses) if all_losses else float('nan')
        out[f'{metric_key_prefix}_max_loss'] = max_loss
        self.model.train()
        return out

args = TrainingArguments(
    output_dir=OUTDIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type='cosine',
    warmup_steps=warmup_steps,
    logging_steps=LOG_STEPS,
    save_steps=save_steps,
    eval_strategy='epoch',
    save_total_limit=8,
    bf16=True,
    report_to=['tensorboard'],
    logging_dir=os.path.join(OUTDIR, 'logs'),
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    load_best_model_at_end=False,
)

trainer = MyTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

print(f'\n{"="*50}')
print(f'TRAINING START')
print(f'Output: {OUTDIR}')
print(f'Train: {len(train_samples)} samples | Test: {len(test_samples)} samples')
print(f'25 epochs × {steps_per_epoch} steps = {total_steps} total')
print(f'Warmup: {warmup_steps} | Save: every {save_steps} steps')
print(f'{"="*50}\n')
trainer.train()

trainer.save_model(OUTDIR)
tokenizer.save_pretrained(OUTDIR)
print(f'\nDone. Model: {OUTDIR}')

