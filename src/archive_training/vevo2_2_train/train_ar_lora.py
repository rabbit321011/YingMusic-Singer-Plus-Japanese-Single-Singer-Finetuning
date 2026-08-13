import json, os, csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

CKPT = '${REMOTE_ROOT}/ckpts/Vevo2/contentstyle_modeling/posttrained'
DATA_DIR = '${REMOTE_ROOT}/final_sum_large'
OUTDIR = '${REMOTE_ROOT}/ckpts/ar_lora'
CACHE_DIR = '${REMOTE_ROOT}/ckpts/ar_ablations/_cache'
MAX_SEQ_LEN = 2048
N_GPU = 6

EPOCHS = 50
LR = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
LOG_STEPS = 50
WARMUP_RATIO = 0.04
SAVE_EPOCHS = 5

os.makedirs(OUTDIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT = 'User will provide you with a text. Please vocalize it with natural expression.'

with open(os.path.join(DATA_DIR, 'train_singnet.json')) as f:
    train_raw = json.load(f)
with open(os.path.join(DATA_DIR, 'test_singnet.json')) as f:
    test_raw = json.load(f)

codes_dir = os.path.join(DATA_DIR, 'codes')

def gen_sample(rec):
    stem = os.path.splitext(os.path.basename(rec['Path']))[0]
    code_path = os.path.join(codes_dir, f'{stem}.pt')
    if not os.path.exists(code_path):
        return None
    codes = torch.load(code_path, map_location='cpu', weights_only=True)
    return rec['Text'], codes['cs_ids'].tolist(), codes['prosody_ids'].tolist()

def build_training_text(text, prosody_ids, cs_ids):
    prosody_str = '<|prosody_start|>' + ''.join(f'<|prosody_{int(i)}|>' for i in prosody_ids) + '<|prosody_end|>'
    cs_str = '<|content_style_start|>' + ''.join(f'<|content_style_{int(i)}|>' for i in cs_ids) + '<|content_style_end|>'
    return (
        f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
        f'<|im_start|>user\n{text}<|im_end|>\n'
        f'<|im_start|>assistant\n{prosody_str}{cs_str}<|im_end|>'
    )

def tokenize_samples(samples, desc='Tokenizing'):
    ids_list, labs_list = [], []
    for text, cs_ids, prosody_ids in tqdm(samples, desc=desc):
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

cache_path = os.path.join(CACHE_DIR, 'tokenized.pt')
if os.path.exists(cache_path):
    print('Loading cached tokenized data...')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    train_ids, train_labs = cache['train_ids'], cache['train_labs']
    test_ids, test_labs = cache['test_ids'], cache['test_labs']
    n_train, n_test = cache['n_train'], cache['n_test']
else:
    train_samples = [s for rec in tqdm(train_raw, desc='Loading train') if (s := gen_sample(rec))]
    test_samples = [s for rec in tqdm(test_raw, desc='Loading test') if (s := gen_sample(rec))]
    n_train, n_test = len(train_samples), len(test_samples)
    train_ids, train_labs = tokenize_samples(train_samples, 'Tokenizing train')
    test_ids, test_labs = tokenize_samples(test_samples, 'Tokenizing test')
    torch.save({
        'train_ids': train_ids, 'train_labs': train_labs,
        'test_ids': test_ids, 'test_labs': test_labs,
        'n_train': n_train, 'n_test': n_test,
    }, cache_path)
    print(f'Cached tokenized data -> {cache_path} (train={n_train}, test={n_test})')

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

CSV_PATH = os.path.join(OUTDIR, 'loss.csv')

class CsvLogger(TrainerCallback):
    def __init__(self):
        with open(CSV_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'epoch', 'train_loss', 'eval_loss'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            with open(CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([state.global_step, round(state.epoch or 0, 2),
                                        logs['loss'], ''])

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs and 'eval_loss' in logs:
            with open(CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([state.global_step, round(state.epoch or 0, 2),
                                        '', logs['eval_loss']])

model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

effective_batch = BATCH_SIZE * GRAD_ACCUM * N_GPU
steps_per_epoch = (n_train + effective_batch - 1) // effective_batch
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
save_steps = steps_per_epoch * SAVE_EPOCHS

print(f'\n{"="*60}')
print(f'  LoRA FINETUNE — r=16, targets=q/k/v/o_proj')
print(f'  Train={n_train}  Test={n_test}')
print(f'  Eff batch={effective_batch} (bs={BATCH_SIZE}*ga={GRAD_ACCUM}*{N_GPU}gpu)')
print(f'  LR={LR:.2e}  Epochs={EPOCHS}')
print(f'  Steps/epoch={steps_per_epoch}  Total={total_steps}  Warmup={warmup_steps}')
print(f'  Save every {SAVE_EPOCHS} epochs = {save_steps} steps')
print(f'  Output: {OUTDIR}')
print(f'{"="*60}\n')

training_args = TrainingArguments(
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
    save_total_limit=15,
    bf16=True,
    report_to=['tensorboard'],
    logging_dir=os.path.join(OUTDIR, 'logs'),
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    load_best_model_at_end=False,
    run_name='lora_r16',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ArDataset(train_ids, train_labs),
    eval_dataset=ArDataset(test_ids, test_labs),
    data_collator=collate_fn,
    callbacks=[CsvLogger()],
)

trainer.train()

model.save_pretrained(os.path.join(OUTDIR, 'final_lora'))
tokenizer.save_pretrained(os.path.join(OUTDIR, 'final_lora'))
print(f'\n=== LoRA DONE. Adapters saved to {OUTDIR}/final_lora ===\n')

