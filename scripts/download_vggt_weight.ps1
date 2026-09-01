param(
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\models\VGGT-1B"),
    [int]$Connections = 16
)

$ErrorActionPreference = "Stop"
$Url = "https://huggingface.co/facebook/VGGT-1B/resolve/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt"
$ExpectedBytes = [int64]5026874952
$ExpectedSha256 = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$PartDir = Join-Path $OutputDir ".parts"
$Target = Join-Path $OutputDir "model.pt"

New-Item -ItemType Directory -Force -Path $PartDir | Out-Null
if (Test-Path -LiteralPath $Target) {
    $existing = Get-Item -LiteralPath $Target
    if ($existing.Length -eq $ExpectedBytes -and
        (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant() -eq $ExpectedSha256) {
        Write-Output "VGGT_READY=$Target"
        exit 0
    }
    Remove-Item -LiteralPath $Target
}

$Connections = [Math]::Max(1, $Connections)
$Chunk = [int64][Math]::Ceiling($ExpectedBytes / $Connections)
$jobs = @()
for ($index = 0; $index -lt $Connections; $index++) {
    $start = [int64]$index * $Chunk
    $end = [Math]::Min($ExpectedBytes - 1, $start + $Chunk - 1)
    if ($start -gt $end) { break }
    $part = Join-Path $PartDir ("model.pt.part{0:D2}" -f $index)
    $expectedPartBytes = $end - $start + 1
    if ((Test-Path -LiteralPath $part) -and (Get-Item -LiteralPath $part).Length -eq $expectedPartBytes) { continue }
    Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
    $jobs += Start-Job -ArgumentList $Url, $start, $end, $part -ScriptBlock {
        param($jobUrl, $jobStart, $jobEnd, $jobPart)
        & curl.exe -L --fail --silent --show-error --retry 5 --retry-all-errors --range "${jobStart}-${jobEnd}" --output $jobPart $jobUrl
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
if ($jobs.Count -gt 0) {
    $jobs | Wait-Job | Out-Null
    $failed = $jobs | Where-Object { $_.State -ne "Completed" }
    $jobs | Receive-Job -ErrorAction Continue | Write-Output
    $jobs | Remove-Job
    if ($failed) { throw "One or more VGGT download segments failed." }
}

$stream = [IO.File]::Open($Target, [IO.FileMode]::Create, [IO.FileAccess]::Write)
try {
    for ($index = 0; $index -lt $Connections; $index++) {
        $part = Join-Path $PartDir ("model.pt.part{0:D2}" -f $index)
        $source = [IO.File]::OpenRead($part)
        try { $source.CopyTo($stream) } finally { $source.Dispose() }
    }
} finally { $stream.Dispose() }

$actual = Get-FileHash -LiteralPath $Target -Algorithm SHA256
if ((Get-Item -LiteralPath $Target).Length -ne $ExpectedBytes -or
    $actual.Hash.ToLowerInvariant() -ne $ExpectedSha256) {
    Remove-Item -LiteralPath $Target -Force -ErrorAction SilentlyContinue
    throw "VGGT checksum verification failed."
}
Remove-Item -LiteralPath $PartDir -Recurse -Force
Write-Output "VGGT_READY=$Target"
