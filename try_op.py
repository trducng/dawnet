import time

from accelerate import init_empty_weights
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    set_seed,
    pipeline,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

import torch
from torch import Tensor
from torch.nn import Parameter

from dawnet import Inspector, op as dop

model_id = "openai-community/gpt2"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
)

text = "Then, Peter and Paul went to the meeting room. Peter gave a key to"
input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

inspector = Inspector(model)
op1_id = inspector.add_op("transformer.h.0.attn", dop.CacheModuleInputOutput())
op2_id = inspector.add_op("transformer.h.0", dop.CacheModuleInputOutput(no_input=True))
op3_id = inspector.add_op("transformer.h.1.attn", dop.SetBreakpoint(
    filename="/home/john/miniconda3/envs/dawnet/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py",
    lineno=507,
))

# inspector.remove_op(op1_id)

attention_mask = torch.ones(input_ids.shape, device=inspector._model.device)
with torch.no_grad():
    output = inspector._model.generate(
        input_ids, attention_mask=attention_mask, max_length=50
    )
    state = inspector.state
    print(len(state.output))

