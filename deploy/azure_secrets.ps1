# Copies the keys from your local .env into the Azure Container App as encrypted secrets,
# so you never type them on the command line (they would end up in PowerShell history).
# Run from the folder with run.py:   powershell -ExecutionPolicy Bypass -File deploy\azure_secrets.ps1
param(
  [string]$App = "floodsight",
  [string]$Group = "floodsight-rg"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".env")) { throw "No .env here. Run this from the folder that contains run.py." }

# .env name -> Azure secret name (secret names: lowercase letters, digits, '-')
$map = [ordered]@{
  "SH_CLIENT_ID" = "sh-client-id"; "SH_CLIENT_SECRET" = "sh-client-secret"
  "CDSE_USERNAME" = "cdse-username"; "CDSE_PASSWORD" = "cdse-password"
  "OHSOME_API_KEY" = "ohsome-api-key"; "FLOODSIGHT_ACCESS_CODE" = "floodsight-access-code"
}
$vals = @{}
foreach ($line in Get-Content ".env") {
  if ($line -match '^\s*([A-Z_]+)\s*=\s*(.*)\s*$') { $vals[$Matches[1]] = $Matches[2].Trim('"').Trim("'") }
}
if (-not $vals["FLOODSIGHT_ACCESS_CODE"]) { throw "Add FLOODSIGHT_ACCESS_CODE=<a long code> to .env first (judges type it to start new analyses)." }

$secrets = @(); $envs = @()
foreach ($k in $map.Keys) {
  if ($vals[$k]) { $secrets += "$($map[$k])=$($vals[$k])"; $envs += "$k=secretref:$($map[$k])" }
  else { Write-Host "  (not in .env, skipped: $k)" }
}
az containerapp secret set -n $App -g $Group --secrets $secrets --output none
az containerapp update -n $App -g $Group --set-env-vars $envs --output none
Write-Host "Done: $($secrets.Count) secrets set (values not shown)."
