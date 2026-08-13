import json, os, sys, csv, argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers import TrainingArguments, Trainer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--grad_accum', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--log_steps', type=int, default=50)
args = parser.parse_args()

CKPT = '${REMOTE_ROOT}/ckpts/Vevo2/contentstyle_modeling/posttrained'
DATA_DIR = '${REMOTE_ROOT}/final_sum_large'
BASE_OUT = '${REMOTE_ROOT}/ckpts/ar_ablations'
OUTDIR = os.path.join(BASE_OUT, args.run_name)
CACHE_DIR = os.path.join(BASE_OUT, '_cache')
MAX_SEQ_LEN = 2048
WARMUP_RATIO = 0.04
N_GPU = 6

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

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
            csv.writer(f).writerow(['run_name', 'step', 'epoch', 'train_loss', 'eval_loss'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            with open(CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([args.run_name, state.global_step, round(state.epoch or 0, 2),
                                        logs['loss'], ''])

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs and 'eval_loss' in logs:
            with open(CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([args.run_name, state.global_step, round(state.epoch or 0, 2),
                                        '', logs['eval_loss']])

model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16)

effective_batch = args.batch_size * args.grad_accum * N_GPU
steps_per_epoch = (n_train + effective_batch - 1) // effective_batch
total_steps = steps_per_epoch * args.epochs
warmup_steps = int(total_steps * WARMUP_RATIO)

print(f'\n{"="*60}')
print(f'  EXPERIMENT: {args.run_name}')
print(f'  Train={n_train}  Test={n_test}')
print(f'  Eff batch={effective_batch} (bs={args.batch_size}*ga={args.grad_accum}*{N_GPU}gpu)')
print(f'  LR={args.lr:.2e}  Epochs={args.epochs}')
print(f'  Steps/epoch={steps_per_epoch}  Total={total_steps}  Warmup={warmup_steps}')
print(f'  Output: {OUTDIR}')
print(f'{"="*60}\n')

training_args = TrainingArguments(
    output_dir=OUTDIR,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    lr_scheduler_type='cosine',
    warmup_steps=warmup_steps,
    logging_steps=args.log_steps,
    save_strategy='no',
    eval_strategy='epoch',
    bf16=True,
    report_to=['tensorboard'],
    logging_dir=os.path.join(OUTDIR, 'logs'),
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    load_best_model_at_end=False,
    run_name=args.run_name,
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
print(f'\n=== {args.run_name} DONE ===\n')

