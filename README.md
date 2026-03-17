# LG Aimers 8 EXAONE 4.0 Quantization and Lightweighting Competition

## 프로젝트 개요

**LG Aimers 8 EXAONE 4.0 관련 대회**에서 진행한 양자화 및 경량화 실험을 정리한 기록입니다.  
최종적으로 사용한 제출 코드와, 해당 코드에 도달하기 전까지 수행했던 시행착오 및 실패 원인을 함께 정리했습니다.

대회 환경에서 실제로 어떤 시도가 유효했고, 어떤 시도가 성능 저하로 이어졌는지를 실험 중심으로 기록하는 데 의미를 두었습니다.

**최종 순위 28/612Team**

**최종 score : 0.63281**

---

## 대회/배경

최근 AI 서비스는 클라우드 기반의 대규모 모델을 넘어, On-Device 환경에서도 빠르고 안정적으로 동작하는 경량 모델에 대한 요구가 급격히 증가하고 있습니다.  
특히 응답 지연(latency), 메모리 사용량, 운영 비용 등의 제약으로 인해 모델 크기를 줄이면서도 성능을 유지하는 기술이 중요한 과제로 부상하고 있습니다.

EXAONE은 Global Frontier 급의 Large-scale 모델과 함께, 노트북·모바일 등 제한된 환경에서도 활용 가능한 Small-scale 모델 라인업을 보유하고 있습니다.  
그러나 단순히 파라미터 수를 줄이는 방식은 메모리와 속도 측면에서는 유리할 수 있으나, 정확도 저하라는 명확한 한계를 동반합니다.

이에 따라, 모델 크기를 효과적으로 축소하면서도 성능 저하를 최소화하고, 실제 추론 환경에서의 효율을 극대화할 수 있는 경량화 기법을 탐구하는 것이 중요했습니다.

이번 해커톤은 이러한 문제의식을 바탕으로, **EXAONE 4.0 모델을 대상으로 한 실전 중심의 LLM 경량화**를 수행하는 대회였습니다.

---

## 평가 리더보드

리더보드 점수는 **성능(Performance)** 과 **속도(Speed)** 를 함께 반영하는 방식으로 계산되었습니다.

### 평가 산식

<img width="802" height="465" alt="image" src="https://github.com/user-attachments/assets/026f3d61-06dd-45c3-8618-d67e91fd7fcd" />

# LG Aimers 8 EXAONE 4.0 Quantization and Lightweighting Competition

## 프로젝트 개요

**LG Aimers 8 EXAONE 4.0 관련 대회**에서 진행한 양자화 및 경량화 실험을 정리한 기록입니다.  
최종적으로 사용한 제출 코드와, 해당 코드에 도달하기 전까지 수행했던 시행착오 및 실패 원인을 함께 정리했습니다.

대회 환경에서 실제로 어떤 시도가 유효했고, 어떤 시도가 성능 저하로 이어졌는지를 실험 중심으로 기록하는 데 의미를 두었습니다.

**최종 순위 28/612Team**

**최종 score : 0.63281**

---

## 대회/배경

최근 AI 서비스는 클라우드 기반의 대규모 모델을 넘어, On-Device 환경에서도 빠르고 안정적으로 동작하는 경량 모델에 대한 요구가 급격히 증가하고 있습니다.  
특히 응답 지연(latency), 메모리 사용량, 운영 비용 등의 제약으로 인해 모델 크기를 줄이면서도 성능을 유지하는 기술이 중요한 과제로 부상하고 있습니다.

EXAONE은 Global Frontier 급의 Large-scale 모델과 함께, 노트북·모바일 등 제한된 환경에서도 활용 가능한 Small-scale 모델 라인업을 보유하고 있습니다.  
그러나 단순히 파라미터 수를 줄이는 방식은 메모리와 속도 측면에서는 유리할 수 있으나, 정확도 저하라는 명확한 한계를 동반합니다.

이에 따라, 모델 크기를 효과적으로 축소하면서도 성능 저하를 최소화하고, 실제 추론 환경에서의 효율을 극대화할 수 있는 경량화 기법을 탐구하는 것이 중요했습니다.

이번 해커톤은 이러한 문제의식을 바탕으로, **EXAONE 4.0 모델을 대상으로 한 실전 중심의 LLM 경량화**를 수행하는 대회였습니다.

---

## 평가 리더보드

리더보드 점수는 **성능(Performance)** 과 **속도(Speed)** 를 함께 반영하는 방식으로 계산되었습니다.

### 평가 산식

<img width="802" height="465" alt="image" src="https://github.com/user-attachments/assets/026f3d61-06dd-45c3-8618-d67e91fd7fcd" />

---

## 최종 코드 상세 설명

### 1. 라이브러리 import

```python
import json
import shutil
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
```

---

### 주요 코드
- **AutoModelForCausalLM**: causal language modeling용 모델을 자동으로 불러옵니다. EXAONE 4.0 기반 원본 모델을 로드하는 데 사용했습니다.
       
- **oneshot**: llmcompressor에서 제공하는 함수로, calibration dataset을 기반으로 모델 양자화를 한 번에 수행하는 역할을 했습니다.

- **QuantizationModifier**:양자화 설정을 정의하는 객체입니다. 어떤 계층을 양자화할지, 어떤 계층은 제외할지, 어떤 quantization scheme을 적용할지를 여기서 지정했습니다.

---

### 2. Calibration 샘플 정의
```base_calib_samples = [ {"messages": [{"role": "user", "content": "단계별로 풀고 마지막 줄에 답만 써줘: 어떤 수의 30%는 45이다. 그 수는?"}]},
{"messages": [{"role": "user", "content": "다음 두 식을 연립하여 x, y를 구하시오. x + y = 10, x - y = 2"}]},
{"messages": [{"role": "user", "content": "이순신 장군이 거북선을 만든 이유를 논리적으로 설명해줘."}]},
{"messages": [{"role": "user", "content": "주어진 JSON 데이터를 파싱하여 Python 딕셔너리로 변환하는 코드를 작성해."}]} ]
```
- **GPTQ 계열 또는 W8A8 계열 양자화에서는 단순히 모델 weight만 바꾸는 것이 아니라, 일정한 입력 데이터를 모델에 통과시켜 보면서 어떤 activation 분포가 나오는지,
  어떤 계층이 민감한지, 어느 부분을 더 잘 보존해야 하는지를 파악해야 했습니다. 이렇게 구성한 이유는 calibration 입력이 한 가지 유형에만 치우치지 않도록 하기 위함이었습니다.
  즉, 짧은 계산 문제, 논리적 설명, 코드 생성 등 서로 다른 특성을 가진 입력을 섞어서 모델이 여러 스타일의 입력 분포를 최소한이라도 보게 만들고자 했습니다.**

---

### 3. Calibration 데이터 확장 및 JSONL 저장
``` calib_data = base_calib_samples * 64 calib_path = ROOT / "calib_optimized.jsonl" with open(calib_path, "w", encoding="utf-8") as f:
    for item in calib_data: f.write(json.dumps(item, ensure_ascii=False) + "\n") print(f"[2] 캘리브레이션 데이터(다양성 확보) 생성 완료: {calib_path}")
```
- **GPTQ 계열 또는 W8A8 계열 양자화에서는 단순히 모델 weight만 바꾸는 것이 아니라, 일정한 입력 데이터를 모델에 통과시켜 보면서 어떤 activation 분포가 나오는지,
  어떤 계층이 민감한지, 어느 부분을 더 잘 보존해야 하는지를 파악해야 했습니다. 이렇게 구성한 이유는 calibration 입력이 한 가지 유형에만 치우치지 않도록 하기 위함이었습니다.
  즉, 짧은 계산 문제, 논리적 설명, 코드 생성 등 서로 다른 특성을 가진 입력을 섞어서 모델이 여러 스타일의 입력 분포를 최소한이라도 보게 만들고자 했습니다.**

---

### 4. Tokenizer와 원본 모델 로드

```tokenizer = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True, local_files_only=True) model = AutoModelForCausalLM.from_pretrained( str(BASE), device_map="cuda:0", dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True ).eval() print(f"[3] 원본 모델 로드 완료 (dtype: {model.dtype})")```


