param([switch]$FetchModels, [switch]$Signed)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
function Run-Checked {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}
if (-not [Environment]::Is64BitProcess -or [Environment]::OSVersion.Platform -ne "Win32NT") {
    throw "Use a Windows x64 build environment."
}
Run-Checked "python" @("-c", "import sys; assert sys.version_info[:2] == (3, 11), 'Use Python 3.11'")
Run-Checked "python" @("-m", "pip", "install", "--require-hashes", "-r", "desktop/requirements.lock", "-r", "desktop/build-requirements.lock")
Run-Checked "npm.cmd" @("ci", "--prefix", "frontend")
Run-Checked "npm.cmd" @("run", "build", "--prefix", "frontend")
Run-Checked "npm.cmd" @("ci", "--prefix", "desktop")
$prepare = @("scripts/prepare-desktop.py")
if ($FetchModels) { $prepare += "--fetch-models" }
Run-Checked "python" $prepare
Run-Checked "python" @("-m", "unittest", "discover", "-s", "tests", "-v")
Run-Checked "python" @("-m", "PyInstaller", "--noconfirm", "--distpath", "desktop", "--workpath", "desktop/build", "desktop/backend.spec")
Run-Checked "$root/desktop/payload/waveatlas-backend.exe" @("--workspace", "$root/desktop/build/check-workspace", "--check")
Run-Checked "python" @("scripts/smoke-desktop.py", "--backend", "$root/desktop/payload/waveatlas-backend.exe")
if ($Signed) {
    if (-not $env:WAVEATLAS_SIGN_COMMAND) { throw "Set WAVEATLAS_SIGN_COMMAND to a signing wrapper that accepts one file path, signs and timestamps it, and fails on errors." }
    Run-Checked $env:WAVEATLAS_SIGN_COMMAND @("$root/desktop/payload/waveatlas-backend.exe")
    $config = @{ bundle = @{ windows = @{ signCommand = @{ cmd = $env:WAVEATLAS_SIGN_COMMAND; args = @("%1") } } } }
    $config | ConvertTo-Json -Depth 6 | Set-Content "desktop/build/signing.json" -Encoding utf8
    Run-Checked "npm.cmd" @("run", "build", "--prefix", "desktop", "--", "--config", "$root/desktop/build/signing.json", "--", "--locked")
} else {
    Run-Checked "npm.cmd" @("run", "build", "--prefix", "desktop", "--", "--", "--locked")
}
$installers = Get-ChildItem "desktop/src-tauri/target/release/bundle/nsis/*.exe"
if ($installers.Count -eq 0) { throw "No Windows installer produced" }
if ($Signed) {
    foreach ($file in @("desktop/payload/waveatlas-backend.exe", "desktop/src-tauri/target/release/waveatlas-desktop.exe") + @($installers.FullName)) {
        if ((Get-AuthenticodeSignature $file).Status -ne "Valid") { throw "Invalid signature: $file" }
    }
}
$installers | ForEach-Object { "$((Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower())  $($_.Name)" } | Set-Content "desktop/src-tauri/target/release/bundle/nsis/SHA256SUMS.txt" -Encoding ascii
Write-Output "Windows installer ready under desktop/src-tauri/target/release/bundle/nsis. Signing requested: $Signed"
