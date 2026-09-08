#==========================================#
# 커밋 전 용량 점검
#   git add 한 다음, push 하기 전에 실행할 것.
#
#   python scripts/check_size.py
#
# GitHub 제한:  50 MiB 경고 / 100 MiB 푸시 거부 / 레포 1GB 권장
#==========================================#
import subprocess
import sys
from pathlib import Path

WARN  = 50 * 1024**2      # GitHub 경고
BLOCK = 100 * 1024**2     # GitHub 푸시 거부
ROOT  = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8").stdout.splitlines()


def human(n):
    return f"{n/1024**2:.1f} MB"


# 스테이징된 파일 + 이미 추적 중인 파일
files = sorted(set(git("ls-files")) | set(git("diff", "--cached", "--name-only")))
if not files:
    print("git 저장소가 아니거나 추적 중인 파일이 없다.")
    sys.exit(0)

total, blocked, warned = 0, [], []
for f in files:
    p = ROOT / f
    if not p.is_file():
        continue
    n = p.stat().st_size
    total += n
    if n > BLOCK:
        blocked.append((f, n))
    elif n > WARN:
        warned.append((f, n))

print(f"추적 파일 {len(files)}개 / 합계 {human(total)}")

if blocked:
    print(f"\n[!!] 100MB 초과 — 이대로 push하면 GitHub이 거부한다 ({len(blocked)}개)")
    for f, n in sorted(blocked, key=lambda x: -x[1]):
        print(f"     {human(n):>10}  {f}")
    print("\n     해결: git rm --cached <파일>  후 .gitignore에 추가하거나,")
    print("           float16으로 저장해 용량을 줄이거나, Git LFS를 쓸 것.")

if warned:
    print(f"\n[! ] 50MB 초과 — push는 되지만 경고가 뜬다 ({len(warned)}개)")
    for f, n in sorted(warned, key=lambda x: -x[1]):
        print(f"     {human(n):>10}  {f}")

if total > 1024**3:
    print(f"\n[! ] 레포 합계가 1GB를 넘었다 ({human(total)}). clone이 느려진다.")

if not blocked and not warned:
    print("\n[ok] 문제 없음. push해도 된다.")

sys.exit(1 if blocked else 0)
