# LG Aimers 8기 EXAONE 4.0 양자화/경량화 실험 기록

## 프로젝트 개요

이 저장소는 **LG Aimers 8기 EXAONE 4.0 관련 대회**에서 진행한 양자화 및 경량화 실험을 정리한 기록이다.  
최종적으로 사용한 제출 코드와, 그 코드에 도달하기 전까지 수행한 시행착오 및 실패 원인을 함께 정리했다.

이 프로젝트의 목적은 단순히 양자화 코드를 보관하는 것이 아니라,  
대회 환경에서 실제로 어떤 시도가 유효했고 어떤 시도가 성능 저하로 이어졌는지를 기록하는 데 있다.

---

## 대회/배경

대회에서는 EXAONE 4.0 기반 모델을 대상으로 양자화 및 경량화를 적용하여 제출해야 했고,  
성능 평가는 정확도뿐 아니라 속도까지 함께 반영되는 구조였다.

실험 과정에서 확인한 핵심 포인트는 다음과 같았다.

- 양자화 오차만 줄인다고 점수가 반드시 오르지 않음
- 대회 서버의 실행 환경(vLLM 및 블랙박스 평가 분포)을 고려해야 함
- calibration 데이터의 분포 변화가 결과에 큰 영향을 줌
- 일부 기법은 개념적으로 우수해도 대회 서버 구조상 적용이 어려움

---

## 최종 접근 방식 요약

최종 제출 코드는 다음 방향을 기준으로 정리했다.

- `llmcompressor`의 `QuantizationModifier` 사용
- `Linear` 계층을 대상으로 `W8A8` 양자화 적용
- `lm_head`는 ignore 처리
- calibration 데이터는 별도의 복잡한 필터링 없이 직접 구성한 샘플을 반복하여 사용
- tokenizer의 chat template을 적용해 calibration text를 구성
- 양자화 완료 후 모델과 tokenizer를 `submit/model`에 저장
- generation_config는 deterministic inference를 위해 일부 항목 수정

최종 코드는 이전 실험들에서 나타난 아래 문제들을 피하는 방향으로 정리되었다.

- 과도한 필터링으로 calibration 분포를 바꾸는 문제
- 임의의 config 수정으로 vLLM 메타데이터 불일치가 발생하는 문제
- 커스텀 forward 경로가 필요한 방식(SmoothQuant 계열)을 대회 서버에 억지로 맞추는 문제
- 속도와 정확도 균형을 해치는 과도한 보호(ignore) 전략

---

## 최종 코드 설명

최종 제출 코드는 `final_quantize.py`에 정리되어 있다.

주요 흐름은 다음과 같다.

1. `base_model` 경로에서 원본 모델 로드
2. calibration용 JSONL 데이터 생성
3. tokenizer의 chat template 적용
4. `QuantizationModifier`로 `W8A8` 양자화 수행
5. 결과 모델과 tokenizer 저장
6. `generation_config.json` 수정

---

## 실행 흐름

### 1. 기본 폴더 준비

- `base_model/` : 원본 모델 디렉토리
- `submit/model/` : 양자화 결과 저장 경로

### 2. 스크립트 실행

`final_quantize.py`를 실행하면 다음이 순서대로 수행된다.

- calibration 데이터 생성
- 원본 모델 로드
- dataset 변환
- W8A8 양자화
- 결과 저장
- generation_config 수정

---

## 폴더 구조

```text
Lg-aimers-8-exaone4-quantization-competition/
├─ README.md
├─ final_quantize.py
├─ docs/
│  ├─ experiment_log.md
│  └─ paper_review.md
├─ outputs/
│  └─ .gitkeep
└─ submit/
   └─ .gitkeep
