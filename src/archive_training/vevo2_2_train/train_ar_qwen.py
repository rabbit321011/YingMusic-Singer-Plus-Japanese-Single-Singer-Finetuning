import json, os, sys, re
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import accelerate

CKPT = '${REMOTE_ROOT}/ckpts/Vevo2/contentstyle_modeling/posttrained'
SUMDIR = '${REMOTE_ROOT}/target singer_hareru_singer_sum'
OUTDIR = '${REMOTE_ROOT}/ckpts/ar_target singer_v2'
MAX_STEPS = 50000
LR = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LEN = 2048
SAVE_STEPS = 2000
LOG_STEPS = 100
EVAL_STEPS = 500

os.makedirs(OUTDIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open(os.path.join(SUMDIR, 'singnet.json'), 'r') as f:
    records = json.load(f)

with open(os.path.join(SUMDIR, 'train_test_split.json'), 'r') as f:
    split = json.load(f)
train_bv = set(split['train_bids'])
test_bv = set(split['test_bids'])

def bv_id(path):
    m = re.match(r'(BV[^_]+)_', os.path.basename(path))
    return m.group(1) if m else ''

codes_dir = os.path.join(SUMDIR, 'codes')

def gen_sample(rec):
    text = rec['Text']
    stem = os.path.splitext(os.path.basename(rec['Path']))[0]
    code_path = os.path.join(codes_dir, f'{stem}.pt')
    if not os.path.exists(code_path):
        return None
    codes = torch.load(code_path, map_location='cpu', weights_only=True)
    cs_ids = codes['cs_ids'].tolist()
    prosody_ids = codes['prosody_ids'].tolist()
    return text, cs_ids, prosody_ids

train_samples = []
val_samples = []
for rec in tqdm(records, desc='Loading samples'):
    s = gen_sample(rec)
    if not s:
        continue
    bid = bv_id(rec['Path'])
    if bid in test_bv:
        val_samples.append(s)
    else:
        train_samples.append(s)

print(f'Train: {len(train_samples)}, Val: {len(val_samples)}')

SYSTEM_PROMPT = 'User will provide you with a text. Please vocalize it with natural expression.'

def build_training_text(text, prosody_ids, cs_ids):
    prosody_str = ''.join([f'<|prosody_{int(i)}|>' for i in prosody_ids])
    prosody_str = '<|prosody_start|>' + prosody_str + '<|prosody_end|>'
    cs_str = ''.join([f'<|content_style_{int(i)}|>' for i in cs_ids])
    cs_str = '<|content_style_start|>' + cs_str + '<|content_style_end|>'
    full = (
        f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
        f'<|im_start|>user\n{text}<|im_end|>\n'
        f'<|im_start|>assistant\n{prosody_str}{cs_str}<|im_end|>'
    )
    return full

class ArDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.input_ids_list = []
        self.labels_list = []
        self.attention_mask_list = []
        for text, cs_ids, prosody_ids in tqdm(samples, desc='Tokenizing'):
            full_text = build_training_text(text, prosody_ids, cs_ids)
            ids = tokenizer.encode(full_text, add_special_tokens=False)
            if len(ids) > max_len:
                ids = ids[:max_len]
            labels = [-100] * len(ids)
            assistant_start_str = '<|im_start|>assistant\n'
            assistant_start_ids = tokenizer.encode(assistant_start_str, add_special_tokens=False)
            for i in range(len(ids) - len(assistant_start_ids) + 1):
                if ids[i:i+len(assistant_start_ids)] == assistant_start_ids:
                    for j in range(i + len(assistant_start_ids), len(ids)):
                        labels[j] = ids[j]
                    break
            self.input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            self.labels_list.append(torch.tensor(labels, dtype=torch.long))
            self.attention_mask_list.append(torch.ones(len(ids), dtype=torch.long))

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'labels': self.labels_list[idx],
            'attention_mask': self.attention_mask_list[idx],
        }

def collate_fn(batch):
    max_len = max(x['input_ids'].size(0) for x in batch)
    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []
    for x in batch:
        pad_len = max_len - x['input_ids'].size(0)
        padded_input_ids.append(torch.cat([x['input_ids'], torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([x['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
        padded_attention_mask.append(torch.cat([x['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels),
        'attention_mask': torch.stack(padded_attention_mask),
    }

train_dataset = ArDataset(train_samples, tokenizer, MAX_SEQ_LEN)
val_dataset = ArDataset(val_samples, tokenizer, MAX_SEQ_LEN)
print(f'Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}')

model = AutoModelForCausalLM.from_pretrained(
    CKPT,
    torch_dtype=torch.bfloat16,
)
model.train()

args = TrainingArguments(
    output_dir=OUTDIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    learning_rate=LR,
    lr_scheduler_type='cosine',
    warmup_steps=2000,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy='steps',
    save_total_limit=5,
    bf16=True,
    report_to=['tensorboard'],
    logging_dir=os.path.join(OUTDIR, 'logs'),
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    load_best_model_at_end=False,
    metric_for_best_model='eval_loss',
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

print(f'Training: {MAX_STEPS} steps, {BATCH_SIZE * GRAD_ACCUM} batch, lr={LR}')
print(f'Train loss reported every {LOG_STEPS} steps, Eval loss every {EVAL_STEPS} steps')
trainer.train()

trainer.save_model(OUTDIR)
tokenizer.save_pretrained(OUTDIR)
print(f'Model saved to {OUTDIR}')

