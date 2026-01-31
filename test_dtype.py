import requests, json
r = requests.get("http://localhost:8880/health")
print(json.dumps(r.json(), indent=2))

# Check model dtype via a debug endpoint or directly
import torch
from qwen_tts import Qwen3TTSModel

# Load and check
model = Qwen3TTSModel.from_pretrained(
    "models/CustomVoice",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
param = next(model.model.parameters())
print(f"Model dtype: {param.dtype}")
print(f"Model device: {param.device}")
print(f"Attn impl: {model.model.config._attn_implementation if hasattr(model.model.config, '_attn_implementation') else 'unknown'}")
