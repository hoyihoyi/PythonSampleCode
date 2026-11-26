# Batch backtests of SessionBreakout on Pepperstone tick data via cTrader CLI.
# Run on the Windows PC that has cTrader installed:
#     powershell -ExecutionPolicy Bypass -File run_backtests.ps1 -Account 1234567 -CtId you@example.com
#
# NOTE: flag names follow the cTrader CLI docs as best known; check them first with
#     & $Cli help backtest
# and adjust below if your version differs. Your password is read from a local file
# (-PwdFile) and never leaves this machine.
param(
    [Parameter(Mandatory = $true)][string]$Account,
    [Parameter(Mandatory = $true)][string]$CtId,
    [string]$PwdFile = "$env:USERPROFILE\ctrader_pwd.txt",
    [string]$Cli = "$env:LOCALAPPDATA\Programs\cTrader CLI\ctrader-cli.exe",
    [string]$Algo = "$PSScriptRoot\SessionBreakout.algo",
    [string]$Start = "01/01/2015",
    [string]$End = (Get-Date -Format "dd/MM/yyyy"),
    [string]$OutDir = "$PSScriptRoot\..\results\ctrader"
)

$symbols = "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD"
$variants = @(
    @{ name = "base";   args = @("--NarrowOnly=false") },
    @{ name = "narrow"; args = @("--NarrowOnly=true") }
)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

foreach ($s in $symbols) {
    foreach ($v in $variants) {
        $tag = "SBO_{0}_{1}" -f $s, $v.name
        Write-Host "== $tag"
        & $Cli backtest $Algo `
            --ctid=$CtId --pwd-file=$PwdFile --account=$Account `
            --symbol=$s --period=h1 --start=$Start --end=$End `
            --data-mode=ticks --balance=10000 `
            --report="$OutDir\$tag.html" --report-json="$OutDir\$tag.json" `
            --exit-on-stop @($v.args)
    }
}
Write-Host "Done. Reports in $OutDir"
