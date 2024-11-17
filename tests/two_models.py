import time

from accelerate import init_empty_weights
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, pipeline
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

import torch
from torch import Tensor
from torch.nn import Parameter


from dawnet.model import ModelRunner

model_id = "openai-community/gpt2"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
)
from copy import deepcopy, _deepcopy_dispatch, _deepcopy_atomic
# _deepcopy_dispatch[Tensor] = _deepcopy_atomic
# _deepcopy_dispatch[Parameter] = _deepcopy_atomic
model2 = deepcopy(model)
import pdb; pdb.set_trace()

text = "Then, Peter and Paul went to the meeting room. Peter gave a key to"
input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    set_seed(42)
    # output = model.generate(input_ids, max_length=50)
    # print(tokenizer.decode(output.squeeze().cpu().numpy().tolist()))
    # generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # set_seed(42)
    # print(generator("Hello I'm a language model,", max_length=50))

def set_parameter(model, path, value):
    paths = path.split(".")
    obj = model
    for path in paths[:-1]:
        if path.isdigit():
            obj = obj[int(path)]
        else:
            obj = getattr(obj, path)

    if paths[-1].isdigit():
        obj[int(paths[-1])] = value
    else:
        setattr(obj, paths[-1], value)

    return model


with init_empty_weights():
    config = AutoConfig.from_pretrained(model_id)
    model3 = GPT2LMHeadModel(config).eval()

with torch.no_grad():
    # for name, param in model.state_dict().items():
    #     set_parameter(model3, name, Parameter(param))
    model3.load_state_dict(model.state_dict(), assign=True)
    attention_mask = torch.ones(input_ids.shape, device=model3.device)
    model3.config.do_sample = True
    set_seed(42)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    output = model3.generate(input_ids, attention_mask=attention_mask, max_length=50)
    print(
        tokenizer.decode(
            output.squeeze().cpu().numpy().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )
    # generator = pipeline("text-generation", model=model3, tokenizer=tokenizer, device="cuda")
    # set_seed(42)
    # print(generator("Then, Peter and Paul went to the meeting room. Peter gave a key to", max_length=50))

    # output = generator.model.generate(input_ids, attention_mask=attention_mask, max_length=50)
    # print(
    #     tokenizer.decode(
    #         output.squeeze().cpu().numpy().tolist(),
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=True,
    #     )
    # )
