# shared/ — 두 노트북이 나눠 쓰는 산출물

**다시 만들려면 오래 걸리는 것**만 여기 넣는다. 이건 git으로 동기화된다.

들어갈 것
- feature 캐시 (예: `cifar10_resnet18_fp16.npz`) — GPU 노트북에서 10분 걸려 뽑은 걸
  gram에서 또 뽑을 이유가 없다
- 전처리 끝낸 중간 데이터

들어가면 안 되는 것
- 원본 데이터셋 (MNIST, CIFAR-10) → `data/`로. 한 줄이면 다시 받는다
- 100MB 넘는 파일 → GitHub이 push를 거부한다

**저장은 float16으로.** float32의 절반이고 feature 캐시 용도로는 정밀도 차이가 없다.

| 백본 | 차원 | train fp32 | train fp16 |
|---|---:|---:|---:|
| ResNet18 | 512 | 97.7 MB (위험) | **48.8 MB** |
| EfficientNet-B0 | 1280 | 244 MB (불가) | 122 MB (불가) |
| ResNet50 | 2048 | 391 MB (불가) | 195 MB (불가) |

그래서 002 CNN은 **ResNet18 + fp16**으로 간다. 50MB 밑으로 들어와서 LFS 없이 그냥 커밋된다.
