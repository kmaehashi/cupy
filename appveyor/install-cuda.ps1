function Download ($filename, $url) {
    $webclient = New-Object System.Net.WebClient

    $basedir = $pwd.Path + "\"
    $filepath = $basedir + $filename
    if (Test-Path $filename) {
        Write-Host "Reusing" $filepath
        return $filepath
    }

    # Download and retry up to 3 times in case of network transient errors.
    Write-Host "Downloading" $filename "from" $url
    $retry_attempts = 2
    for ($i = 0; $i -lt $retry_attempts; $i++) {
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
    }
    if (Test-Path $filepath) {
        Write-Host "File saved at" $filepath
    } else {
        # Retry once to get the error message if any at the last try
        $webclient.DownloadFile($url, $filepath)
    }
    return $filepath
}

function RunCommand ($command, $command_args) {
    Write-Host $command $command_args
    Start-Process -FilePath $command -ArgumentList $command_args -Wait -Passthru
}

function Unzip ([string]$zipfile, [string]$outpath) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

function main () {
    # Download & Install CUDA
    $filename = "cuda_9.1.85_windows.exe"
    $url = "https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_windows"
    $filepath = Download $filename $url
    RunCommand $filepath "-s"
    $cuda_path = "$Env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v9.1"

    # Download & Install cuDNN
    $filename = "cudnn-9.1-windows7-x64-v7.zip"
    $url = "http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/$filename"
    $filepath = Download $filename $url
    Unzip $filepath $pwd.Path
    Move-Item "cuda\bin\cudnn64_7.dll" "$cuda_path\bin"
    Move-Item "cuda\lib\x64\cudnn.lib" "$cuda_path\lib\x64"
    Move-Item "cuda\include\cudnn.h"   "$cuda_path\include"
}

main
