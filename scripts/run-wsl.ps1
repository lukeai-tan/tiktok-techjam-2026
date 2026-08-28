$ErrorActionPreference = "Stop"
$PythonArgs = $args
$repoWindows = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ($repoWindows -notmatch "^(?<drive>[A-Za-z]):(?<tail>.*)$") {
    throw "Expected a drive-letter Windows path, got: $repoWindows"
}
$repoDrive = $Matches.drive.ToLowerInvariant()
$repoTail = $Matches.tail.Replace("\", "/")
$repoWsl = "/mnt/$repoDrive$repoTail"

$wslUser = (wsl -d Ubuntu -- id -un).Trim()
if (-not $wslUser) {
    throw "Could not determine the default Ubuntu user."
}
$wslPython = $env:TIKTOK_TECHJAM_PYTHON
if (-not $wslPython) {
    $wslPython = "/home/$wslUser/.venvs/tiktok-techjam-2026/bin/python"
}

$compiler = "$repoWsl/tools/triton-cc"
wsl -d Ubuntu --cd $repoWsl -- env `
    "PATH=/home/$wslUser/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" `
    "CC=$compiler" `
    "PYTHONPATH=." `
    $wslPython @PythonArgs
exit $LASTEXITCODE
