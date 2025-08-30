param(
    [string]$Configuration = "Debug",
    [string]$Platform = "x64"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $root "VestaEngine.vcxproj"
$defaultMsBuild = Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe"

if (Test-Path $defaultMsBuild) {
    $msbuild = $defaultMsBuild
} else {
    $msbuildCommand = Get-Command MSBuild.exe -ErrorAction Stop
    $msbuild = $msbuildCommand.Source
}

& $msbuild $project /t:CustomBuild /p:Configuration=$Configuration /p:Platform=$Platform /v:minimal /nologo
exit $LASTEXITCODE
