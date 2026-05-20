param(
    # 作用：服务监听地址。默认仅本机访问，避免 0.0.0.0 的权限问题。
    [string]$BindHost = "127.0.0.1",

    # 作用：服务监听端口。默认 18001，与项目文档保持一致。
    [int]$Port = 18001,

    # 作用：等待服务启动的超时时间（秒）。超时会自动报错并输出日志。
    [int]$StartupTimeoutSec = 120,

    # 作用：端口被占用时是否自动顺延寻找可用端口（默认开启，避免手工改端口）。
    [bool]$AutoPickFreePort = $true,

    # 作用：自动找端口时最多尝试多少个连续端口。
    [int]$MaxPortSearch = 20,

    # 作用：默认测试完成后自动停止服务；传入该参数可让服务保留运行。
    [switch]$KeepService
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# 作用：定位项目根目录（脚本所在目录），避免在错误目录执行。
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# 作用：定位项目虚拟环境 Python，确保使用固定依赖运行。
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonPath)) {
    throw "未找到虚拟环境 Python：$PythonPath。请先完成环境准备。"
}

# 作用：示例请求文件路径。
$ExamplePath = Join-Path $ProjectRoot "example_preprocess_request.json"
if (-not (Test-Path $ExamplePath)) {
    throw "未找到示例请求文件：$ExamplePath"
}

# 作用：检查指定端口是否已有监听进程（仅检查 LISTEN 状态）。
function Get-ListeningConnection {
    param(
        [int]$TargetPort
    )

    Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
}

# 作用：从起始端口开始，顺延查找第一个可用端口；找不到时返回 $null。
function Find-AvailablePort {
    param(
        [int]$StartPort,
        [int]$MaxTry
    )

    for ($offset = 0; $offset -le $MaxTry; $offset++) {
        $candidate = $StartPort + $offset
        $listening = Get-ListeningConnection -TargetPort $candidate
        if ($null -eq $listening) {
            return $candidate
        }
    }

    return $null
}

# 作用：优先使用用户传入端口；若被占用且启用自动切换，则自动选择可用端口。
$occupied = Get-ListeningConnection -TargetPort $Port
if ($null -ne $occupied) {
    $owner = $occupied.OwningProcess

    if (-not $AutoPickFreePort) {
        throw "端口 $Port 已被占用（PID=$owner）。请先释放端口，或改用 -Port 参数。"
    }

    $nextPort = Find-AvailablePort -StartPort ($Port + 1) -MaxTry $MaxPortSearch
    if ($null -eq $nextPort) {
        throw "端口 $Port 已被占用（PID=$owner），且在后续 $MaxPortSearch 个端口内未找到可用端口。请手动指定 -Port。"
    }

    $ownerProc = Get-Process -Id $owner -ErrorAction SilentlyContinue | Select-Object -First 1
    $ownerName = if ($null -ne $ownerProc) { $ownerProc.ProcessName } else { "unknown" }
    Write-Host "端口 $Port 已被占用（PID=$owner, Name=$ownerName），自动切换到端口 $nextPort。"
    $Port = $nextPort
}

# 作用：设置运行环境变量，保证 Windows 本地稳定测试。
$env:HOME = $ProjectRoot
$env:USERPROFILE = $ProjectRoot
$env:DESENSITIZE_STRICT_LOCAL_MODEL = "true"
$env:DESENSITIZE_ENABLE_TASKFLOW = "true"
$env:DESENSITIZE_AUTO_DOWNLOAD_MODEL = "false"
$env:HOST = $BindHost
$env:PORT = "$Port"

# 作用：保存服务日志，便于失败时快速定位。
$StdOutLog = Join-Path $ProjectRoot "run_service.out.log"
$StdErrLog = Join-Path $ProjectRoot "run_service.err.log"
if (Test-Path $StdOutLog) { Remove-Item -LiteralPath $StdOutLog -Force }
if (Test-Path $StdErrLog) { Remove-Item -LiteralPath $StdErrLog -Force }

# 作用：服务进程退出后保存 stdout/stderr，便于失败时定位。
function Save-ProcessOutput {
    param(
        [System.Diagnostics.Process]$Process
    )

    if ($null -eq $Process -or -not $Process.HasExited) {
        return
    }

    try {
        $stdout = $Process.StandardOutput.ReadToEnd()
        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            Set-Content -LiteralPath $StdOutLog -Value $stdout -Encoding UTF8
        }
    }
    catch {
    }

    try {
        $stderr = $Process.StandardError.ReadToEnd()
        if (-not [string]::IsNullOrWhiteSpace($stderr)) {
            Set-Content -LiteralPath $StdErrLog -Value $stderr -Encoding UTF8
        }
    }
    catch {
    }
}

Write-Host "[1/4] 启动服务进程（$BindHost`:$Port）..."
$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $PythonPath
$startInfo.Arguments = "app.py"
$startInfo.WorkingDirectory = $ProjectRoot
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$proc = [System.Diagnostics.Process]::new()
$proc.StartInfo = $startInfo
[void]$proc.Start()

try {
    $healthUrl = "http://$BindHost`:$Port/healthz"
    $apiUrl = "http://$BindHost`:$Port/v1/llm/preprocess"

    Write-Host "[2/4] 等待服务就绪：$healthUrl"
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSec)
    $ready = $false

    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500

        if ($proc.HasExited) {
            Save-ProcessOutput -Process $proc
            $stderrTail = if (Test-Path $StdErrLog) { (Get-Content $StdErrLog -Tail 80 | Out-String) } else { "<empty>" }
            $stdoutTail = if (Test-Path $StdOutLog) { (Get-Content $StdOutLog -Tail 80 | Out-String) } else { "<empty>" }
            throw "服务进程已提前退出。`n---STDERR---`n$stderrTail`n---STDOUT---`n$stdoutTail"
        }

        try {
            $health = Invoke-RestMethod -Method Get -Uri $healthUrl -TimeoutSec 2
            if ($health.status -eq "ok") {
                $ready = $true
                break
            }
        }
        catch {
            # 启动阶段允许短暂连接失败，继续重试。
        }
    }

    if (-not $ready) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force
            $proc.WaitForExit()
        }
        Save-ProcessOutput -Process $proc
        $stderrTail = if (Test-Path $StdErrLog) { (Get-Content $StdErrLog -Tail 80 | Out-String) } else { "<empty>" }
        $stdoutTail = if (Test-Path $StdOutLog) { (Get-Content $StdOutLog -Tail 80 | Out-String) } else { "<empty>" }
        throw "服务启动超时（${StartupTimeoutSec}s）。`n---STDERR---`n$stderrTail`n---STDOUT---`n$stdoutTail"
    }

    Write-Host "[3/4] 服务已就绪，执行示例请求：$apiUrl"
    $body = Get-Content -LiteralPath $ExamplePath -Raw -Encoding UTF8
    $resp = Invoke-RestMethod -Method Post -Uri $apiUrl -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 60

    Write-Host "[4/4] 示例请求执行成功，返回如下："
    $resp | ConvertTo-Json -Depth 20
}
finally {
    if (-not $KeepService) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force
            $proc.WaitForExit()
            Save-ProcessOutput -Process $proc
            Write-Host "已停止服务进程（PID=$($proc.Id)）。"
        }
    }
    else {
        if ($proc -and -not $proc.HasExited) {
            Write-Host "服务保持运行（PID=$($proc.Id)）。"
        }
    }
}
