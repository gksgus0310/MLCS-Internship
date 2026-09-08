# ============================================================
#  MLCS 인턴십 과제 - Windows 환경 자동 세팅
#
#  PowerShell에서 레포 폴더로 이동한 뒤:
#      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#      .\setup.ps1
#
#  주의 1) 파이썬 3.14 에는 파이토치 CUDA 휠이 없다. GPU를 쓰려면 3.12/3.13 필요.
#  주의 2) RTX 50 시리즈(Blackwell, sm_120)는 CUDA 12.8 이상 빌드라야 동작한다.
# ============================================================

$PY_TARGET = "3.12"        # 가상환경에 쓸 파이썬 버전
$CUDA_TAG  = "cu128"       # RTX 50 시리즈용. 구형 GPU면 pytorch.org에서 확인 후 변경

Write-Host "`n[1/5] 파이썬 확인" -ForegroundColor Cyan
$pyOk = $false
try {
    $v = & py -$PY_TARGET --version 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "  $v 사용" -ForegroundColor Green; $pyOk = $true }
} catch {}
if (-Not $pyOk) {
    Write-Host "  파이썬 $PY_TARGET 를 찾을 수 없다." -ForegroundColor Red
    Write-Host "  https://www.python.org/downloads/release/python-3120/ 에서 설치할 것."
    Write-Host "  설치 시 'Add python.exe to PATH' 체크 필수."
    Write-Host "  기존 파이썬은 지우지 않아도 된다 (여러 버전 공존 가능)."
    exit 1
}

Write-Host "`n[2/5] 가상환경 생성 (.venv)" -ForegroundColor Cyan
if (-Not (Test-Path ".venv")) { & py -$PY_TARGET -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip -q
Write-Host "  가상환경 파이썬: $(python --version)"

Write-Host "`n[3/5] GPU 확인" -ForegroundColor Cyan
$hasGpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    $hasGpu = $true
} else {
    Write-Host "  nvidia-smi 없음 -> CPU 전용으로 설치한다." -ForegroundColor Yellow
}

Write-Host "`n[4/5] PyTorch 설치" -ForegroundColor Cyan
if ($hasGpu) {
    Write-Host "  CUDA 빌드 ($CUDA_TAG) 설치 - 2~3GB, 시간이 좀 걸린다"
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
} else {
    Write-Host "  CPU 빌드 설치 - CUDA 빌드보다 2~3GB 가볍다"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "`n[5/5] 나머지 패키지 설치" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n환경 점검" -ForegroundColor Green
python check_env.py
