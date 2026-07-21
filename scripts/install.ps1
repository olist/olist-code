# Installs the latest olist-code binary release.
#
# Usage:
#   irm https://raw.githubusercontent.com/olist/olist-code/main/scripts/install.ps1 | iex
#
# Requires either the GitHub CLI (gh, already authenticated) or a
# GITHUB_TOKEN env var with read access to olist/olist-code, since this
# is a private repository.

$ErrorActionPreference = "Stop"

$Repo = "olist/olist-code"
$Asset = "olist-code-windows-x86_64.exe"
$InstallDir = if ($env:OLIST_CODE_INSTALL_DIR) { $env:OLIST_CODE_INSTALL_DIR } else { "$env:LOCALAPPDATA\olist-code\bin" }
$Dest = Join-Path $InstallDir "olist-code.exe"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

if (Get-Command gh -ErrorAction SilentlyContinue) {
    Write-Host "Downloading $Asset via gh release download..."
    gh release download --repo $Repo --pattern $Asset --output $Dest --clobber
}
elseif ($env:GITHUB_TOKEN) {
    Write-Host "Downloading $Asset via GitHub API..."
    $headers = @{
        Authorization = "Bearer $env:GITHUB_TOKEN"
        Accept        = "application/vnd.github+json"
    }
    $release = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$Repo/releases/latest"
    $assetInfo = $release.assets | Where-Object { $_.name -eq $Asset }
    if (-not $assetInfo) {
        throw "Could not find asset $Asset in the latest release."
    }
    $downloadHeaders = @{
        Authorization = "Bearer $env:GITHUB_TOKEN"
        Accept        = "application/octet-stream"
    }
    Invoke-WebRequest -Headers $downloadHeaders -Uri $assetInfo.url -OutFile $Dest
}
else {
    throw "Need either the 'gh' CLI (authenticated) or a GITHUB_TOKEN env var to download from the private repo."
}

Write-Host "Installed olist-code to $Dest"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$InstallDir*") {
    Write-Host "Add it to your PATH: [Environment]::SetEnvironmentVariable('Path', `"`$env:Path;$InstallDir`", 'User')"
}
