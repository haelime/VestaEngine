# VestaEngine

VestaEngine는 `Vulkan/C++` 기반의 개인 렌더링 엔진 프로젝트다.  
유니티 게임 프로그래머 포지션 관점에서는 다음 역량을 보여 주기 위해 만든 작업물이다.

- 렌더링 파이프라인을 실제 실행 가능한 코드로 구조화하는 능력
- 디버그 UI, 카메라 툴, 벤치마크 로그처럼 개발 생산성을 높이는 런타임 도구 설계 능력
- Scene streaming, GPU upload, pass orchestration처럼 게임 런타임과 가까운 시스템 작업 능력
- 그래픽스 버그를 수학, 데이터 경로, 셰이더, GPU 작업 흐름 단위로 분해해 수정하는 능력

## 프로젝트 요약

- 멀티패스 프레임 구조: `Geometry -> Deferred Lighting -> Gaussian -> Path Tracing -> Composite`
- 공식 3DGS 스타일 Gaussian 데이터 로드와 compute 기반 tile raster 경로 구현
- glTF/GLB scene loading, async parse/prepare, streaming GPU upload
- ImGui 기반 runtime tuning, benchmark overlay, recent scene UX
- Path tracing backend `Auto / Compute / Hardware RT`

## 포트폴리오 관점 핵심 성과

- `official Gaussian renderer pipeline`을 엔진 구조 안에 통합했다.
- Gaussian 수학 경로를 다시 검토해 `rotation / scale / covariance` 계산 오류와 데이터 손실 경로를 수정했다.
- 디버그 UI를 정리해 `Stats / Render / Camera` 중심으로 재구성하고, 카메라 위치/회전 직접 입력을 추가했다.
- Gaussian pass 최적화로 `garden_official`, `1700x900`, `RTX 5060 Ti`, `Gaussian mode`, `Compute backend` 기준:
  - 평균 프레임 시간 `84~86ms -> 58.38ms`
  - 평균 FPS `11.7~11.9 -> 17.13`
  - padded duplicate count `16,777,216 -> 13,850,397`

## 내가 해결한 대표 문제

### 1. 점구름처럼 보이는 Gaussian을 공식 3DGS에 가까운 경로로 끌어올리기

- point sprite 기반 경로에서 시작하지 않고 `preprocess -> duplicate -> scan -> radix sort -> tile range -> raster` 흐름으로 재구성했다.
- 공식 학습 결과 폴더 구조를 직접 읽고, `position + scale + rotation + opacity + SH coefficients`를 엔진 데이터 구조로 옮겼다.

### 2. 눈에 보이는 artifact를 수학/데이터 문제로 분해해 수정하기

- rotation이 반영되지 않거나 scale이 과도하게 잘리는 문제를 셰이더와 scene import 양쪽에서 다시 검토했다.
- opacity floor, anisotropy cap, 잘못된 covariance 전개 같은 데이터 손실 경로를 제거했다.

### 3. 개발자 입장에서 필요한 디버그 UI를 직접 설계하기

- `Scene` 창을 `Stats`에 통합하고, `Detailed Info` 토글로 핵심 정보와 상세 정보를 분리했다.
- 카메라 위치/회전을 직접 입력할 수 있게 해 반복 검증과 비교가 쉬운 형태로 바꿨다.

### 4. 병목을 기능이 아니라 비용 구조 관점에서 줄이기

- duplicate capacity가 한 번 커진 뒤 계속 남아 있던 구조를 정리했다.
- sort/range pass가 실제 작업량이 아니라 capacity 기준으로 과하게 dispatch되던 부분을 `DispatchIndirect` 기반으로 바꿨다.

## Unity 게임 프로그래머 포지션과의 연결

- `렌더링 시스템 이해`: Unity SRP, custom render feature, post-process/debug pass를 다룰 때 필요한 사고방식을 직접 구현으로 검증했다.
- `툴/디버그 역량`: 인게임 디버그 UI, benchmark overlay, camera controls는 Unity Editor tooling과 runtime debug UI 작업으로 바로 전이 가능하다.
- `성능 최적화`: 프레임 타임, duplicate count, upload timing처럼 측정 가능한 지표를 기준으로 문제를 줄이는 접근을 사용했다.
- `콘텐츠 파이프라인 감각`: scene loading, GPU upload, asset path handling은 실제 게임 런타임 자원 관리와 맞닿아 있다.
- `문제 해결 방식`: 현상을 눈으로만 보지 않고, 데이터 구조, 셰이더 수식, GPU dispatch 크기까지 내려가서 원인을 분리했다.

## 문서 가이드

- 대표 포트폴리오 문서: [docs/unity-game-programmer-portfolio-vestaengine.md](docs/unity-game-programmer-portfolio-vestaengine.md)
- Gaussian 시행착오 회고: [docs/gaussian-retrospective.md](docs/gaussian-retrospective.md)
- 공식 3DGS 대비 기술 분석: [docs/gaussian-official-repo-gap-analysis.md](docs/gaussian-official-repo-gap-analysis.md)

## Build

### CMake

```powershell
cmake -S . -B build
cmake --build build --target engine --config Debug
```

### Visual Studio

Open [VestaEngine.sln](VestaEngine.sln) or [VestaEngine.vcxproj](VestaEngine.vcxproj) and build `Debug|x64`.

## Run

### Default

```powershell
.\x64\Debug\VestaEngine.exe
```

### Load a specific scene

```powershell
.\x64\Debug\VestaEngine.exe --scene assets\structure.glb --preset balanced --mode composite
```

### Automated benchmark

```powershell
.\x64\Debug\VestaEngine.exe --scene assets\structure.glb --benchmark out\benchmark.csv --warmup-seconds 2 --benchmark-seconds 10 --preset quality --mode composite --pt-backend auto --no-ui
```

Benchmark CSV에는 프레임 타임뿐 아니라 scene stats, upload timing, active backend, Gaussian-specific metadata가 함께 기록된다.

## Controls

- `RMB + Mouse`: look
- `W/A/S/D/Q/E`: move
- `Shift`: faster move
- `1/2/3/4`: Composite / Deferred / Gaussian / Path Trace
- `G`: toggle Gaussian
- `P`: toggle Path Tracing
- `F1`: toggle debug overlay

## 현재 범위와 한계

- 이 프로젝트는 `학습된 3D Gaussian 결과를 소비하고 렌더링하는 엔진`에 가깝다.
- 공식 3DGS 전체 범위인 `학습 / densification / pruning / 평가 파이프라인`까지 포함하는 프로젝트는 아니다.
- 포트폴리오 관점에서는 대신 `실시간 렌더링 구조`, `디버그 툴`, `그래픽스 버그 수정`, `성능 최적화` 역량을 명확히 보여 주는 데 집중했다.

## References

- Official repository and pre-trained models entry: https://github.com/graphdeco-inria/gaussian-splatting
- Official project page: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
