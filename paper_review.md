# 논문 리뷰

이 문서는 LG Aimers 8기 EXAONE 4 양자화 대회 실험과 연결되는 관점에서 참고한 논문들을 요약한 기록입니다. 각 논문의 핵심 포인트를 짧게 정리하였습니다.

## GPTQ
- 핵심 포인트: Calibration 데이터를 이용한 Hessian 근사로 FP weight를 저비트로 변환하는 양자화 기법.
- 대회 연결: 기본 양자화 방식으로 사용, calibration 샘플 수와 범위 조정 실험에 적용.

## SmoothQuant
- 핵심 포인트: Activation scale 적용으로 양자화 오차를 줄이는 기법, weight와 activation 재스케일링 필요.
- 대회 연결: 데이콘 서버에서 커스텀 연산 불가능하여 적용 실패, activation 재스케일링 개념 참고.

## llm.int8()
- 핵심 포인트: 8비트 양자화로 메모리 절약, outlier 처리 기법.
- 대회 연결: W8A8 양자화 실험에 적용, 안정성 확인.

## Distillation + Quantization 관련 논문
- 핵심 포인트: 지식 증류와 양자화를 결합하여 성능 유지.
- 대회 연결: 양자화 시 정확도 저하 방지를 위한 개념 참고.

## EXAONE 4.0 논문
- 핵심 포인트: EXAONE 4.0 모델의 아키텍처 설명.
- 대회 연결: 모델 구조 이해를 위한 기반, 양자화 적용 시 레이어별 특성 고려.

## EXAONE 4.0 멀티 어텐션 + 멀티 마스크 디코더 논문
- 핵심 포인트: 멀티 어텐션과 마스크 디코더 구조.
- 대회 연결: 모델의 어텐션 메커니즘 이해, 양자화 시 outlier 발생 위치 분석에 활용.

## AWQ
- 핵심 포인트: Activation-aware weight quantization으로 outlier 보호.
- 대회 연결: GPTQ와 결합 실험 시도, 누적 양자화 실패 원인 분석.