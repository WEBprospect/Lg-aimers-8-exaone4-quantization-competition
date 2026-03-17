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

```
tokenizer = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True, local_files_only=True) model = AutoModelForCausalLM.from_pretrained( str(BASE), device_map="cuda:0", dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True ).eval() print(f"[3] 원본 모델 로드 완료 (dtype: {model.dtype})")
```

- **dtype=torch.bfloat16 모델을 bfloat16 정밀도로 로드했습니다. 이는 FP32보다 메모리 사용량을 줄이면서도 비교적 안정적인 정밀도를 제공하는 형식입니다. trust_remote_code=True 해당 모델이 Hugging Face 기본 클래스 외에 커스텀 tokenizer 또는 모델 구현을 포함하고 있을 수 있기 때문에 사용했습니다. EXAONE 계열처럼 custom code가 필요한 경우 이 옵션이 중요할 수 있습니다.
local_files_only=True 인터넷에서 다시 다운로드하지 않고, 로컬에 이미 저장된 파일만 사용하도록 강제합니다. 대회 제출용 또는 오프라인 실험 환경에서 재현성을 높이는 데 도움이 됩니다.**

---

### 5.Calibration dataset 변환 및 Quantization 설정 정의

```
ds = load_dataset("json", data_files=str(calib_path))["train"]
ds = ds.map( lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}, remove_columns=ds.column_names )
```
- **이 부분은 저장해 둔 calibration .jsonl 파일을 dataset으로 불러오고, 모델 입력 형식에 맞게 text로 변환한 뒤, 실제 양자화 설정을 정의하는 단계였습니다.**

- **targets=["Linear"]**: 모델 내부의 Linear 계층들을 양자화 대상으로 지정했습니다. Transformer 모델의 주요 projection 계층들은 대부분 Linear이므로, 이 설정은 실질적으로 핵심 weight들을 양자화하는 의미를 가집니다..
       
- **ignore=["lm_head"]**: 최종 출력층인 lm_head는 양자화 대상에서 제외했습니다. 출력 직전 계층은 민감도가 높을 수 있으므로, 이를 제외해 성능 저하를 줄이려는 의도가 반영된 설정입니다.

- **scheme="W8A8"**:weight와 activation을 모두 8비트 기준으로 다루는 W8A8 양자화 방식을 사용했습니다. 이는 W4보다 일반적으로 양자화 오차가 작고, 성능 안정성을 확보하기에 유리한 선택이었습니다.

---

### 6. 양자화 수행 및 모델 저장

```
print("[4] 모델 양자화(W8A8) 시작..") oneshot( model=model, recipe=recipe, dataset=ds, max_seq_length=512, num_calibration_samples=len(calib_data), ) model.save_pretrained(str(OUT_DIR)) tokenizer.save_pretrained(str(OUT_DIR))
```
- **max_seq_length=512 calibration 시 고려할 최대 시퀀스 길이를 512로 제한했습니다. 너무 긴 문맥까지 포함하면 계산량이 커질 수 있으므로, 실험 기준에서 적절한 길이로 제한한 것입니다. num_calibration_samples=len(calib_data) calibration 샘플 수를 전체 calib_data 길이와 동일하게 설정했습니다. 이 코드에서는 총 256개 샘플을 사용합니다.**

---

## 최종 코드가 적절했던 이유

이 최종 코드는 복잡한 기법을 많이 얹기보다, **대회 환경에서 실제로 안정적으로 동작할 수 있는 양자화 파이프라인**에 집중했다는 점에서 의미가 있었습니다.

첫째, **실행 환경과의 호환성을 우선적으로 고려했습니다.**  
실험 과정에서 SmoothQuant 계열처럼 개념적으로는 우수하지만, 커스텀 forward 경로나 추가적인 activation rescaling이 필요한 방식은 대회 서버 환경과 맞지 않는다는 점을 확인했습니다.  
반면 최종 코드는 `llmcompressor`의 기본 양자화 흐름을 사용하여, 제출 환경에서 비교적 재현 가능하고 안정적으로 동작할 수 있는 형태로 정리했습니다.

둘째, **calibration 분포를 과도하게 왜곡하지 않았습니다.**  
이전 실험에서는 `last_user_only`, 과도한 길이 필터링, shuffle, seed 변화 등이 calibration 결과를 크게 흔들 수 있다는 점을 확인했습니다.  
최종 코드에서는 복잡한 전처리 대신, 직접 구성한 대표 샘플을 반복하여 사용함으로써 calibration 입력을 단순하고 일관되게 유지했습니다.  
이는 블랙박스 평가 환경에서 불필요한 분포 미스매치를 줄이는 데 도움이 되었습니다.

셋째, **정확도와 속도의 균형을 고려한 설정을 사용했습니다.**  
이 대회는 성능만이 아니라 속도도 함께 평가되었기 때문에, 일부 레이어를 과도하게 보호하거나 복잡한 혼합 정밀도 전략을 쓰면 오히려 전체 점수가 하락할 수 있었습니다.  
최종 코드의 `W8A8` 설정은 지나치게 공격적인 저비트 양자화보다 안정적이면서도, 속도 측면에서도 균형을 맞추기 좋은 선택이었습니다.

넷째, **출력층 민감도를 고려해 `lm_head`를 제외했습니다.**  
출력 직전 계층은 작은 오차도 최종 토큰 확률에 직접 영향을 줄 수 있기 때문에, 이를 양자화 대상에서 제외한 것은 성능 저하를 줄이기 위한 현실적인 선택이었습니다.

다섯째, **제출 파이프라인까지 포함한 완결된 코드였습니다.**  
이 코드는 단순히 양자화만 수행하는 데서 끝나지 않고, calibration 데이터 생성, 모델 로드, dataset 변환, 양자화 실행, 결과 저장, 그리고 `generation_config.json` 수정까지 하나의 흐름으로 정리되어 있었습니다.  
즉, 실험용 조각 코드가 아니라 실제 제출을 염두에 둔 재현 가능한 파이프라인이라는 점에서 의미가 있었습니다.

결국 이 최종 코드는 가장 화려한 기법을 사용한 코드는 아니었지만,  
**대회 서버 환경에서 안정적으로 실행될 수 있고, calibration 분포 왜곡을 줄이며, 성능과 속도의 균형을 맞춘 현실적인 해법**이었다는 점에서 좋은 선택이었습니다.
