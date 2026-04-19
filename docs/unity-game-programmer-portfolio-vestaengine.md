# VestaEngine 포트폴리오 정리

`VestaEngine`는 Vulkan/C++ 기반의 개인 렌더링 엔진 프로젝트다.  
유니티 게임 프로그래머 포지션 기준으로는 "그래픽스 지식이 있는 엔진/런타임 프로그래머가 실제 문제를 어떻게 정의하고, 디버깅하고, 성능까지 줄였는가"를 보여 주는 사례로 정리할 수 있다.

관련 문서:

- 저장소 개요: [../README.md](../README.md)
- Gaussian 시행착오 회고: [gaussian-retrospective.md](gaussian-retrospective.md)
- 공식 3DGS 대비 분석: [gaussian-official-repo-gap-analysis.md](gaussian-official-repo-gap-analysis.md)

## 1. 프로젝트 한 줄 요약

멀티패스 렌더링 구조 위에 `Geometry / Deferred / Gaussian / Path Tracing / Composite`를 통합하고, 공식 3DGS 스타일 Gaussian 데이터를 실시간으로 렌더링할 수 있도록 만든 개인 엔진 프로젝트다.

## 2. 이 프로젝트를 포트폴리오로 사용하는 이유

이 프로젝트는 단순히 "그래픽스 공부를 했다"가 아니라 아래 역량을 결과물로 보여 준다.

- 복잡한 렌더링 기능을 pass 단위로 구조화하는 능력
- scene loading, GPU upload, debug UI, benchmark logging 같은 실무형 보조 시스템 설계 능력
- 보이는 artifact를 데이터 경로와 수식 단위까지 내려가서 수정하는 분석 능력
- 프레임 타임과 GPU 작업량을 기준으로 최적화하는 습관

Unity 포지션으로 옮기면 다음과 같이 읽힐 수 있다.

- SRP나 custom render feature를 다룰 수 있는 그래픽스 기반
- 인게임 디버그 UI와 개발자 도구를 직접 설계할 수 있는 툴 감각
- 렌더링/로딩/자원 관리처럼 런타임 시스템을 건드릴 수 있는 엔진 감각
- 현상을 감으로 때우지 않고 계측과 근거로 문제를 줄이는 개발 방식

## 3. 프로젝트에서 맡은 문제 정의

제가 이 프로젝트에서 집중한 문제는 크게 4가지였다.

| 문제 | 의미 |
| --- | --- |
| 공식 3DGS 스타일 Gaussian을 엔진 구조 안에 통합하기 | 단순 점구름이 아니라 학습 결과를 소비하는 렌더러 필요 |
| 디버깅 가능한 UI와 카메라 도구 만들기 | 반복 검증과 비교를 사람이 빠르게 할 수 있어야 함 |
| Gaussian 수학 오류와 데이터 손실 경로 제거하기 | rotation/scale/covariance가 틀리면 결과가 맞아도 신뢰할 수 없음 |
| Gaussian pass 비용 구조 줄이기 | dense scene에서 실시간성이 급격히 무너지는 병목 제거 |

## 4. 개발 흐름 요약

최근 작업 흐름은 아래 커밋들로 정리할 수 있다.

| 커밋 | 의미 |
| --- | --- |
| `2c40495` | threading, scene architecture 정리 |
| `61600cd` | streaming upload 개선 |
| `4b1815d` | official Gaussian renderer pipeline 도입 |
| `39dc286` | ImGui/Camera 개선 + Gaussian projection math 수정 |
| `e2ff0b9` | official Gaussian pass dispatch sizing 최적화 |

이 흐름이 보여 주는 건 "기능 추가 -> 디버그 수단 확보 -> 수학 수정 -> 성능 최적화" 순서로 문제를 다뤘다는 점이다.  
즉 기능만 붙인 것이 아니라, 나중에 유지보수 가능한 상태로 끌고 갔다.

## 5. 대표 사례 1: 공식 Gaussian renderer pipeline 통합

### 문제

초기 접근은 point sprite나 soft point cloud에 가까웠다.  
이 방식은 겉보기에는 그럴듯하지만, 공식 3DGS의 `position / scale / rotation / opacity / SH`를 제대로 반영하지 못한다.

### 내가 한 일

- trained Gaussian 폴더 구조를 직접 읽는 로더를 추가했다.
- Gaussian 데이터를 엔진 내부 CPU/GPU 구조로 변환했다.
- 렌더링 경로를 `preprocess -> duplicate -> scan -> radix sort -> tile range -> raster` 구조로 재구성했다.
- 기존 mesh/deferred/path tracing 흐름과 충돌하지 않도록 composite 단계에 통합했다.

### 의미

이 작업은 단순 셰이더 1개 추가가 아니라, `asset path -> scene data -> render pass -> UI -> benchmark`까지 이어지는 파이프라인 작업이었다.  
Unity 관점에서는 SRP 기능 추가와 그 기능을 프로젝트 전체 흐름에 맞게 넣는 문제에 가깝다.

## 6. 대표 사례 2: 디버그 UI와 카메라 도구 재설계

### 문제

Gaussian은 카메라 각도, depth, view-dependent color에 민감하다.  
그런데 디버그 UI가 흩어져 있거나, 카메라를 수치로 고정하기 어렵다면 비교 검증이 매우 비효율적이다.

### 내가 한 일

- `Scene` 창을 `Stats`로 합치고 핵심 정보만 남겼다.
- `Detailed Info` 토글을 두어 기본 상태와 상세 상태를 분리했다.
- 카메라 위치와 `Yaw / Pitch / Roll`을 직접 입력할 수 있게 했다.
- Gaussian 관련 runtime 상태와 benchmark overlay를 한 화면에서 읽을 수 있게 정리했다.

### 결과

- 같은 장면/같은 시점에서 artifact를 재현하고 비교하기 쉬워졌다.
- 수학 수정 전후 결과를 더 안정적으로 검증할 수 있게 됐다.

### Unity에서의 의미

실무에서는 기능을 만드는 것만큼 "문제를 재현하고 검증할 수 있는 도구를 만드는 능력"이 중요하다.  
이 작업은 Unity Inspector custom tooling, runtime debug window, QA용 reproduction HUD와 유사한 성격을 가진다.

## 7. 대표 사례 3: Gaussian 수학 오류와 데이터 손실 경로 수정

### 문제

렌더링이 얼핏 맞아 보여도 아래 같은 문제가 남아 있었다.

- rotation이 covariance에 제대로 반영되지 않음
- import 단계에서 opacity와 scale이 과하게 잘림
- 좌표계 변환이 position에는 적용되지만 quaternion에는 누락됨
- screen covariance 전개가 카메라 회전과 섞일 때 왜곡될 수 있음

### 내가 한 일

- `transpose(M) * M`처럼 rotation이 지워지는 공분산 계산을 `M * transpose(M)`로 수정했다.
- opacity floor, anisotropy cap 같은 데이터 손실 경로를 제거했다.
- import transform 시 Gaussian quaternion도 함께 변환하도록 고쳤다.
- `J * viewLinear` 기준으로 screen covariance를 다시 계산하도록 셰이더를 수정했다.

### 결과

- anisotropic Gaussian 방향이 더 일관되게 보이게 됐다.
- 원본 데이터가 import 과정에서 덜 손실되도록 정리됐다.
- "그럴듯해 보이는 결과"가 아니라 수학적으로 납득 가능한 경로로 바뀌었다.

### Unity에서의 의미

이 사례는 그래픽스 버그를 단순 shader tweak가 아니라 `데이터 구조 + import path + 수식 + runtime validation` 전체 문제로 보는 태도를 보여 준다.  
Unity에서도 SRP 버그, VFX/ShaderGraph 한계 보완, 좌표계/카메라 문제 디버깅에 그대로 연결된다.

## 8. 대표 사례 4: official Gaussian pass 최적화

### 문제

기능이 맞아도 dense scene에서 비용 구조가 비효율적이면 실시간성이 급격히 떨어진다.  
특히 duplicate capacity가 한 번 커지면 계속 유지되고, sort/range pass가 실제 work count보다 큰 크기로 돌고 있었다.

### 내가 한 일

- duplicate capacity가 과도하게 커진 상태를 계속 끌고 가지 않도록 shrink 조건을 넣었다.
- scan 단계가 이미 계산한 실제 block count를 활용해 sort/range를 `DispatchIndirect`로 실행하게 바꿨다.
- 즉 "기능 유지"보다 "빈 workgroup을 얼마나 덜 돌릴 수 있는가"를 기준으로 비용 구조를 정리했다.

### 수치

같은 `garden_official` 장면, `1700x900`, `Gaussian mode`, `Compute backend`, `RTX 5060 Ti` 기준으로 관찰된 대표 수치는 아래와 같다.

| 항목 | 이전 관찰값 | 이후 관찰값 |
| --- | --- | --- |
| avg frame ms | `84.06 ~ 85.67` | `58.38` |
| avg fps | `11.67 ~ 11.90` | `17.13` |
| gaussian duplicates | `13,850,397` | `13,850,397` |
| gaussian padded duplicates | `16,777,216` | `13,850,397` |
| avg tiles touched | `2.37376` | `2.37376` |

여기서 중요한 건 duplicate 자체보다 `padded duplicate`를 줄였다는 점이다.  
즉 scene 내용은 그대로 두고, 쓸데없이 확보하고 정렬하던 메모리/작업량을 줄였다.

주의할 점도 있다.

- parse/prepare/upload 시간은 각 실행 시점에 따라 편차가 있어 직접 비교 지표로 쓰지 않았다.
- 대신 이 값들을 benchmark CSV에 남기도록 해, 렌더링 성능뿐 아니라 로딩 파이프라인도 추적 가능하게 만들었다.

## 9. 시행착오에서 배운 점

이 프로젝트에서 가장 중요했던 건 "처음부터 정답을 맞힌 것"이 아니라 시행착오를 구조화했다는 점이다.

- point cloud와 trained Gaussian을 같은 문제로 보면 안 된다.
- 화면에서 비슷하게 보여도 covariance 수학이 틀리면 언젠가 깨진다.
- fixed-cap tile 구조는 dense scene에서 artifact를 구조적으로 만든다.
- CPU 전체 sort와 대용량 재업로드는 임시로는 쉬워도 엔진 응답성을 망친다.
- 기능 추가 뒤에는 반드시 검증 도구와 수치 기록이 따라와야 한다.

이 내용은 더 자세히 [gaussian-retrospective.md](gaussian-retrospective.md)에 정리했다.

## 10. Unity 게임 프로그래머 포지션에 어떻게 연결되는가

### 그래픽스

- SRP, custom pass, GPU-driven rendering 개념을 이해하고 실제 구현으로 옮길 수 있다.

### 런타임 시스템

- scene loading, upload budget, pass orchestration처럼 게임 프레임 안에서 일어나는 시스템 문제를 다룰 수 있다.

### 툴과 디버깅

- 디버그 UI, benchmark overlay, reproducible camera setup 같은 개발자 도구를 직접 설계할 수 있다.

### 성능 최적화

- 프레임 타임, duplicate count, upload timing처럼 계측 가능한 지표를 바탕으로 최적화한다.
- “느리다”가 아니라 “어디서 얼마가 낭비되는지”를 먼저 찾는 방식으로 접근한다.

## 11. 솔직한 한계

- 이 프로젝트는 공식 3DGS 전체 시스템이 아니라 `학습된 결과를 소비하는 렌더링 엔진`에 가깝다.
- Unity 프로젝트 자체는 아니기 때문에 `Unity API 실무 경험`을 직접 보여 주는 문서는 아니다.
- 대신 Unity 팀에서도 중요한 `그래픽스 문제 해결력`, `디버깅 도구 설계`, `성능 계측 기반 최적화`, `엔진 구조화` 역량은 분명하게 보여 줄 수 있다.

## 12. 한 문장 정리

이 프로젝트는 "렌더링 기능을 붙여 본 사람"이 아니라,  
`문제를 정의하고 -> 디버그 수단을 만들고 -> 수학을 바로잡고 -> 수치로 최적화까지 한 사람`이라는 점을 보여 주는 포트폴리오다.
