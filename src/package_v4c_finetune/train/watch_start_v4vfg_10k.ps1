param(
    [string]$Server = "${SERVER_HOST}"
)

$ErrorActionPreference = "Stop"
$LocalDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RemoteDir = "${REMOTE_ROOT}/YingMusic-Singer-Plus"
$Session = "v4vfg_10k"
$OutputDir = "$RemoteDir/ckpts/plus_ja_sft_v4vfg_v4vf30k_vae285k_10k"
$LogPath = "$RemoteDir/train_v4vfg_v4vf30k_vae285k_10k.log"
$Files = @("prepare_v4vfg_bootstrap.py", "run_sft_v4vfg_10k.sh")

function Write-Status([string]$Message) {
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

while ($true) {
    & ssh -o BatchMode=yes -o ConnectTimeout=15 $Server "true" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Server unavailable; retrying in 60 seconds"
        Start-Sleep -Seconds 60
        continue
    }

    $check = & ssh -o BatchMode=yes -o ConnectTimeout=15 $Server @"
set -e
if tmux has-session -t '$Session' 2>/dev/null; then echo SESSION_EXISTS; exit 3; fi
if [ -e '$OutputDir' ]; then echo OUTPUT_EXISTS; exit 4; fi
if [ -e '$LogPath' ]; then echo LOG_EXISTS; exit 5; fi
for index in 0 1 2 3; do
  used=`$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v i=`$index '`$1==i {print `$2}')
  if [ -z "`$used" ] || [ "`$used" -gt 500 ]; then echo GPU_BUSY:`$index:`$used; exit 6; fi
done
test -f '$RemoteDir/ckpts/plus_ja_sft_v4vf_lr1e4_30k/step_030000.pt'
test -f '${REMOTE_ROOT}/experiments/vae_full_official_300k_20260716/checkpoints/autoencoder_285k.ckpt'
echo READY
"@
    if ($LASTEXITCODE -eq 6) {
        Write-Status ($check -join " ")
        Start-Sleep -Seconds 60
        continue
    }
    if ($LASTEXITCODE -ne 0 -or ($check -notcontains "READY")) {
        Write-Status "Preflight failed: $($check -join ' ')"
        exit 2
    }

    foreach ($file in $Files) {
        & scp -q -o BatchMode=yes -o ConnectTimeout=15 (Join-Path $LocalDir $file) "${Server}:$RemoteDir/$file"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to upload $file"
        }
    }

    $launch = @"
set -e
cd '$RemoteDir'
python -m py_compile prepare_v4vfg_bootstrap.py
bash -n run_sft_v4vfg_10k.sh
chmod +x run_sft_v4vfg_10k.sh
tmux new-session -d -s '$Session' "export CUDA_VISIBLE_DEVICES=0,1,2,3; bash run_sft_v4vfg_10k.sh 2>&1 | tee '$LogPath'"
sleep 2
tmux has-session -t '$Session'
echo STARTED
"@
    $result = & ssh -o BatchMode=yes -o ConnectTimeout=15 $Server $launch
    if ($LASTEXITCODE -ne 0 -or ($result -notcontains "STARTED")) {
        Write-Status "Launch failed: $($result -join ' ')"
        exit 3
    }

    Write-Status "Started tmux $Session on GPUs 0,1,2,3"
    exit 0
}
