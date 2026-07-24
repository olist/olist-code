# Installs the latest olist-code binary release.
#
# Usage:
#   irm https://raw.githubusercontent.com/olist/olist-code/main/scripts/install.ps1 | iex

$ErrorActionPreference = "Stop"

$Repo = "olist/olist-code"
$Asset = "olist-code-windows-x86_64.exe"
$InstallDir = if ($env:OLIST_CODE_INSTALL_DIR) { $env:OLIST_CODE_INSTALL_DIR } else { "$env:LOCALAPPDATA\olist-code\bin" }
$Dest = Join-Path $InstallDir "olist-code.exe"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Write-Host "Downloading $Asset..."
Invoke-WebRequest -Uri "https://github.com/$Repo/releases/latest/download/$Asset" -OutFile $Dest

Write-Host "Installed olist-code to $Dest"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$InstallDir*") {
    Write-Host "Add it to your PATH: [Environment]::SetEnvironmentVariable('Path', `"`$env:Path;$InstallDir`", 'User')"
}
