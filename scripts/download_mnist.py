#==========================================#
# MNIST 원본 파일 수동 다운로드 (자동 다운로드가 막혔을 때만 필요)
# Usage: python scripts/download_mnist.py
#==========================================#
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "MNIST" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/fgnt/mnist/master"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

for name in FILES:
    dst = RAW / name
    if dst.exists():
        print(f"[skip] {name} ({dst.stat().st_size:,} bytes)")
        continue
    print(f"[get ] {name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(f"{BASE}/{name}", dst)
    print(f"{dst.stat().st_size:,} bytes")

print(f"\n완료: {RAW}")
