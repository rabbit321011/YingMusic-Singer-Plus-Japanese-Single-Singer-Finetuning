$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$blocked = @('.wav','.mp3','.flac','.pt','.pth','.ckpt','.safetensors','.bin','.tar','.gz','.zip','.7z','.sqlite','.db')
$files = Get-ChildItem $root -Recurse -File | Where-Object { $_.FullName -notmatch '\\.git\\' }
$badExt = @($files | Where-Object { $blocked -contains $_.Extension.ToLowerInvariant() })
if ($badExt.Count) { $badExt | ForEach-Object { Write-Error "Blocked file: $($_.FullName)" } }
$patterns = @(
    'BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY',
    '(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret)\s*[:=]\s*[^$<{\s]',
    '(?i)(password|passwd)\s*[:=]\s*[^$<{\s]'
)
foreach ($pattern in $patterns) {
    Push-Location $root
    $hits = rg -n --hidden --glob '!.git/**' --glob '!scripts/public_release_audit.ps1' --pcre2 $pattern . 2>$null
    Pop-Location
    if ($LASTEXITCODE -eq 0) { Write-Error "Sensitive pattern '$pattern' found:`n$($hits -join "`n")" }
}
Write-Output "PASS: $($files.Count) files audited"
exit 0
