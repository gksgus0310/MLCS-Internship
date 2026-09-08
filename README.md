# MLCS 인턴십 과제 — 노트북 2대 세팅

노트북 A(GPU 있음)와 노트북 B(GPU 없음) **양쪽에서 똑같이 돌아가게** 만드는 구성.

## 무엇을 git에 넣고 무엇을 안 넣나

판단 기준은 용량이 아니라 **"다시 만드는 데 얼마나 걸리나"** 다.

| | git 동기화 | 이유 |
|---|:---:|---|
| 코드 `.py` | **O** | 당연히 |
| 리포트 `.md`, 그래프 `.png`, 수치 `.json` | **O** | 작고, 노트북 바꿔가며 봐야 함 |
| 작은 가중치 `.pt` (수 MB) | **O** | 재학습이 아까움. MLP 것은 3MB뿐 |
| **`shared/` — feature 캐시 등 재현 비싼 중간 산출물** | **O** | GPU 노트북에서 10분 걸려 뽑은 걸 gram에서 또 뽑을 이유가 없다 |
| 원본 데이터셋 (MNIST, CIFAR-10) | X | 한 줄이면 다시 받는다. CIFAR-10은 163MB라 **넣고 싶어도 못 넣는다** |
| 가상환경 `.venv`, PyTorch 빌드 | X | 머신마다 달라야 한다 (GPU/CPU) |

### 데이터셋을 git에 안 넣는 이유

넣고 싶어도 막히는 경우가 먼저다. **GitHub은 파일 하나가 100 MiB를 넘으면 push를 거부하고,
50 MiB를 넘으면 경고한다.** 레포 전체는 1GB 미만 권장, 5GB 미만 강력 권장이다.
CIFAR-10 python 버전이 163MB니까 **그냥 튕긴다.**

크기를 떠나서도 바이너리는 git과 궁합이 나쁘다. 텍스트와 달리 델타 압축이 안 돼서
커밋할 때마다 파일 전체가 히스토리에 통째로 쌓이고, **한 번 올리면 나중에 지워도
히스토리에 남아서** clone이 영영 무거워진다. 반면 `datasets.CIFAR10(download=True)` 한 줄이면
어느 노트북에서든 다시 받는다. 재현 가능한 건 저장하는 게 아니라 다시 만드는 게 맞다.

### 대신 `shared/` 를 쓴다

지적한 문제의식은 맞다. **다시 만들기 비싼 건 공유해야 한다.**
그래서 `shared/` 폴더를 뒀고, 여기는 git으로 동기화된다.

002 CNN이 딱 그 경우다. 사전학습 백본으로 CIFAR-10 feature를 뽑는 데 gram에서 10분쯤 걸리는데,
한 번 뽑아두면 헤드만 바꿔가며 실험할 수 있다. 이걸 커밋해두면 다른 노트북에서 `git pull` 한 번으로
그 10분을 건너뛴다. 단, **float16으로 저장**해야 50MB 밑으로 들어온다:

| 백본 | 차원 | train fp32 | train fp16 |
|---|---:|---:|---:|
| **ResNet18** | 512 | 97.7 MB (위험) | **48.8 MB** |
| EfficientNet-B0 | 1280 | 244 MB (불가) | 122 MB (불가) |
| ResNet50 | 2048 | 391 MB (불가) | 195 MB (불가) |

그래서 002는 ResNet18로 간다. LFS 없이 그냥 커밋된다.

### Git LFS는?

100MB 넘는 걸 꼭 git에 넣어야 하면 Git LFS가 정식 해법이다. 다만 GitHub 무료 계정은
LFS 스토리지·대역폭 무료 한도가 작아서 데이터셋을 넣으면 금방 찬다. **지금 구성으로는 필요 없다.**
나중에 정말 필요해지면 그때 붙이면 된다.

### 사고 방지

push 전에 한 번 돌리면 100MB 사고를 미리 잡는다:

```powershell
git add .
python scripts\check_size.py
```

---

## 최초 1회: GitHub 비공개 레포 만들기

GitHub에서 **New repository → Private** 으로 `MLCS-Internship` 생성 (README 체크 해제).
그 다음 이 폴더에서:

```powershell
git init
git branch -M main
git add .
git commit -m "MLCS internship: 001 MLP"
git remote add origin https://github.com/<your-id>/MLCS-Internship.git
git push -u origin main
```

> 원본 연구실 레포는 별도로 놔두는 게 낫다. 과제 자료를 참고할 땐 그냥 따로 clone해두고,
> 내가 푼 답안은 이 개인 레포에서 관리하면 섞이지 않는다.

Windows 2대 조합이면 줄바꿈 설정도 양쪽 다 한 번씩 맞춰둔다:

```powershell
git config --global core.autocrlf true
```

---

## 새 노트북에서 세팅 (두 대 모두 동일)

### 1단계 — 클론

```powershell
cd C:\Users\<사용자>\Documents
git clone https://github.com/<your-id>/MLCS-Internship.git
cd MLCS-Internship
```

### 2단계 — 자동 세팅 (추천)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

`setup.ps1`이 가상환경 생성 → **nvidia-smi로 GPU 유무를 자동 감지** → 맞는 PyTorch 빌드 설치 →
나머지 패키지 설치 → 환경 점검까지 알아서 한다. 노트북 두 대에서 **똑같은 명령**을 쓰면 된다.

### 3단계 — 수동으로 할 경우

`setup.ps1`이 안 먹으면 아래를 직접 친다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**여기가 두 노트북이 갈리는 유일한 지점이다.**

```powershell
# GPU 있는 노트북 (RTX 50 시리즈 = Blackwell 이면 cu128 이상이라야 한다)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# GPU 없는 노트북 (CUDA 빌드보다 2~3GB 가볍다)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **함정 1 - 파이썬 버전.** 파이썬 3.14 에는 파이토치 CUDA 휠이 아직 없다.
> 3.14 에서 `pip install torch` 하면 조용히 **CPU 빌드만** 깔리고 GPU가 놀게 된다.
> 3.12 나 3.13 으로 가상환경을 만들 것. 기존 파이썬을 지울 필요는 없다.
>
> **함정 2 - GPU 세대.** RTX 50 시리즈(Blackwell, sm_120)는 **CUDA 12.8 이상** 빌드라야 돌아간다.
> cu126 이하를 깔면 "no kernel image is available" 에러가 난다.
>
> 그래서 `requirements.txt`에는 torch를 **일부러 넣지 않았다.** 넣으면 두 노트북 중 한쪽이 반드시 깨진다.

```powershell
pip install -r requirements.txt
python check_env.py
```

### 4단계 — 확인

`check_env.py`가 이렇게 나오면 성공이다.

```
machine       DESKTOP-XXXX
torch         2.x.x+cu126     ← GPU 노트북
CUDA available True
GPU           NVIDIA GeForce RTX ...
```

```
machine       LAPTOP-YYYY
torch         2.x.x+cpu       ← CPU 노트북
CUDA available False
GPU           -
```

**둘 다 정상이다.** 코드가 `torch.device("cuda" if torch.cuda.is_available() else "cpu")`로
알아서 갈라지기 때문에 스크립트는 양쪽에서 수정 없이 그대로 돌아간다.

---

## 매일 쓰는 흐름

노트북을 바꿀 때마다 이 두 줄만 지키면 꼬이지 않는다.

```powershell
# 작업 시작할 때
git pull

# 작업 끝내고 노트북 덮기 전에  ← 이걸 빼먹는 게 사고의 99%
git add .
git commit -m "001 MLP: dropout 제거 실험 추가"
git push
```

가상환경 활성화는 매번:

```powershell
.\.venv\Scripts\Activate.ps1
```

`git pull`이 충돌났다면 대개 양쪽에서 같은 파일을 고친 경우다. 당황하지 말고
`git status`로 어떤 파일인지 보고 고르면 된다.

---

## 폴더 구조

```
MLCS-Internship/
├── README.md                  ← 이 문서
├── requirements.txt           ← torch 빼고 공통 패키지
├── setup.ps1                  ← Windows 자동 세팅
├── check_env.py               ← 환경 점검
├── .gitignore                 ← 원본 데이터·venv만 제외
├── data/                      ← 원본 데이터셋 (git 제외, 노트북마다 자동 다운로드)
├── shared/                    ← 재현 비싼 산출물 (git 동기화) ★
│   └── README.md
├── scripts/
│   ├── download_mnist.py      ← 자동 다운로드 막힐 때 수동 받기
│   └── check_size.py          ← push 전 용량 점검
└── 01_Neural_Networks/
    └── 001_MLP/
        ├── mlp_compare.py     ← 과제 코드
        ├── report.md          ← 결과 리포트
        └── results/           ← 그래프·수치·작은 가중치 (커밋됨)
```

폴더 이름에 **띄어쓰기를 안 썼다.** 원본 레포는 `01 Neural Networks`처럼 공백이 있는데,
PowerShell에서 경로마다 따옴표를 씌워야 해서 귀찮고 실수가 잦다.

---

## 실행

```powershell
python 01_Neural_Networks\001_MLP\mlp_compare.py
```

어느 폴더에서 실행하든 상관없다. 스크립트가 `Path(__file__).resolve()` 기준으로 경로를 잡아서
데이터는 항상 레포의 `data/`, 결과는 항상 자기 옆 `results/`에 저장한다.
**절대경로를 코드에 박지 않는 것**이 노트북 두 대를 쓸 때 제일 중요한 규칙이다.

---

## 알아두면 좋은 것

**결과 수치가 두 노트북에서 미세하게 다를 수 있다.** 시드를 고정해도 CPU와 GPU는 부동소수점
연산 순서가 달라서 소수점 아래가 갈린다. 정상이다. 그래서 `summary.json`에 어느 머신에서
뽑았는지 같이 기록하게 해뒀다. 리포트에 넣을 최종 수치는 **한쪽 노트북 것으로 통일**하는 게 깔끔하다.

**원본 데이터는 각 노트북이 처음 실행할 때 한 번씩 받는다.** MNIST 약 12MB, CIFAR-10 약 163MB.
`data/`는 git 제외라 push해도 안 올라간다.

**무거운 계산은 GPU 노트북에서 하고 결과를 `shared/`로 넘긴다.** 코드는 양쪽에서 다 돌아가지만
굳이 같은 계산을 두 번 할 이유는 없다. GPU 노트북에서 feature를 뽑아 `shared/`에 커밋하면,
gram에서는 `git pull` 하고 헤드 학습만 돌리면 된다. 작은 가중치(.pt 수 MB)도 같이 커밋된다.
