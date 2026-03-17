# 최종 양자화 코드

# 이 파일은 EXAONE 4.0 모델에 GPTQ 양자화를 적용하는 최종 코드입니다.
# 실제 코드 내용은 제공된 최종 코드에 따라 삽입되어야 합니다.
# 현재는 placeholder로 두었으며, 사용자가 제공한 코드를 여기에 붙여넣으세요.

# 예시 코드 구조 (실제 코드로 교체 필요)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 모델 로드
model_name = "LGAI-EXAONE/EXAONE-4.0-7.4B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# 양자화 설정
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    damp_percent=0.01,
    block_name_to_quantize=["model.layers.28.mlp.up_proj"]  # 보호 레이어
)

# GPTQ 양자화
quantized_model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
quantized_model.quantize(model, use_triton=False)

# 저장
quantized_model.save_quantized("outputs/quantized_model")