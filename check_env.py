#==========================================#
# Title:  환경 점검 스크립트
#         새 노트북에서 세팅 끝난 뒤 이거부터 실행할 것.
# Usage:  python check_env.py
#==========================================#
import json
import platform
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("=" * 62)
print(f"{'machine':<16}{socket.gethostname()}")
print(f"{'OS':<16}{platform.system()} {platform.release()}")
print(f"{'python':<16}{sys.version.split()[0]}  ({sys.executable})")
print("-" * 62)

info = {
    "hostname": socket.gethostname(),
    "os": f"{platform.system()} {platform.release()}",
    "python": sys.version.split()[0],
}

# ---- PyTorch ----
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda_ok else "-"
    print(f"{'torch':<16}{torch.__version__}")
    print(f"{'CUDA available':<16}{cuda_ok}")
    print(f"{'GPU':<16}{gpu}")
    info.update(torch=torch.__version__, cuda=cuda_ok, gpu=gpu,
                device="cuda" if cuda_ok else "cpu")
    if not cuda_ok and "+cu" in torch.__version__:
        print("  ! CUDA 빌드인데 GPU를 못 찾는다. 드라이버를 확인하거나,")
        print("    GPU 없는 노트북이면 CPU 빌드로 다시 깔면 용량을 크게 아낄 수 있다.")
except ImportError:
    print("torch         >>> 설치 안 됨. README 3단계를 볼 것.")
    info["torch"] = None

# ---- 나머지 패키지 ----
print("-" * 62)
for mod in ["torchvision", "numpy", "matplotlib", "pandas", "scipy",
            "gymnasium", "stable_baselines3", "yfinance", "qpsolvers", "sklearn"]:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"  [o] {mod:<20}{v}")
        info[mod] = v
    except ImportError:
        print(f"  [ ] {mod:<20}(없음)")
        info[mod] = None

# ---- 데이터 폴더 ----
print("-" * 62)
data = ROOT / "data"
print(f"{'data 경로':<14}{data}")
if data.exists():
    sub = sorted(p.name for p in data.iterdir() if p.is_dir())
    print(f"{'받아둔 데이터':<16}{sub if sub else '(아직 없음 — 처음 실행 시 자동 다운로드)'}")

(ROOT / "env_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print("=" * 62)
print("env_info.json 저장 완료 (이 파일은 .gitignore 처리되어 있음)")
