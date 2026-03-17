# LG Aimers 8기 EXAONE 4 양자화 대회

## 프로젝트 개요
이 프로젝트는 LG Aimers 8기 EXAONE 4 양자화 대회에 참가하기 위해 수행된 양자화 실험 및 최종 코드 정리입니다. EXAONE 4.0 모델을 대상으로 양자화 기법을 적용하여 성능과 속도의 균형을 맞추는 것을 목표로 하였습니다.

## 대회/배경
LG Aimers 8기 대회에서 EXAONE 4.0 모델의 양자화를 통해 모델 크기를 줄이고 추론 속도를 향상시키는 과제를 수행하였습니다. 평가 지표는 성능 정규화 점수(PerfNorm)와 속도 정규화 점수(SpeedNorm)의 가중 평균으로 구성됩니다.

## 최종 접근 방식 요약
최종적으로 GPTQ 기반 양자화를 적용하였으며, W4A16 양자화 방식을 채택하여 outlier가 많이 발생하는 특정 레이어(28번째 레이어의 up_proj)를 보호하는 전략을 사용하였습니다. Calibration 데이터는 기본 설정을 유지하고, 샘플 수는 256개로 하였습니다.

## 폴더 구조
- `README.md`: 프로젝트 설명
- `experiment_log.md`: 실험 시행착오 기록
- `paper_review.md`: 참고 논문 리뷰
- `outputs/`: 양자화 결과 출력 폴더 (빈 폴더, 실제 출력 파일은 제외)
- `submit/`: 제출용 모델 폴더 (빈 폴더, 실제 모델 파일은 제외)

## 실험 기록 문서 링크 안내
자세한 실험 시행착오 및 실패 원인은 `experiment_log.md`를 참고하세요.

## 참고한 논문 목록
- GPTQ
- SmoothQuant
- llm.int8()
- Distillation + Quantization 관련 논문
- EXAONE 4.0 논문
- EXAONE 4.0 멀티 어텐션 + 멀티 마스크 디코더 논문
- AWQ

자세한 논문 리뷰는 `paper_review.md`를 참고하세요.