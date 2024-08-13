from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


from dawnet.model import ModelRunner

model_id = "openai-community/gpt2"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)
runner = ModelRunner(model)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


# with torch.no_grad():
#     l = model(input_ids)
#     print(type(l))


def record(r, n, l, ia, ik, o):
    """
    r: the runner
    n: name of the layer
    l: the layer
    ia: the input arguments
    ik: the input keyword arguments
    o: the output
    """
    # r.ctx and r.output contain the context and output of the runner
    print(n)
    return o


def record2(r, n, l, ia, ik, o):
    """
    r: the runner
    n: name of the layer
    l: the layer
    ia: the input arguments
    ik: the input keyword arguments
    o: the output
    """
    # r.ctx and r.output contain the context and output of the runner
    ctx, output = r.ctx, r.output
    var = ctx.get("var", {})
    if o[0].argmax(-1) == 3:
        var[ctx["text"]] = output["layer10[0]"]
        ctx["var"] = var
    return o



handler1 = runner.add_forward_hooks(record, "transformer.h.2")
handler2 = runner.cache_outputs("transformer.h.9", "transformer.h.10")
handler3 = runner.cache_inputs("transformer.h.8", "transformer.h.9")
handler4 = runner.cache_layers("transformer.h.1")
handler5 = runner.add_ctx(var={}, hello=1)
runner.pdb = True
with torch.no_grad(), runner.add_forward_hooks(record, "transformer.h.3"):
    l = runner(input_ids)
    print(type(l))


# for name, module in model.named_modules():
#     if name == "transformer.h.10":
#         xyz = add_hooks_output(name, module)
