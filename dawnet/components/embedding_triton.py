"""https://gist.github.com/mobicham/739625b54c7b65b00529ab8d794272b6"""
import torch
import triton
import triton.language as tl

def get_configs():
    configs = []
    #for (w, s) in [(1, 1), (1, 2), (2, 1), (2, 2), (4, 4)]:
    for (w, s) in [(8, 4)]:
        for e in [64, 128, 256, 512, 1024]:
            for n in [64, 128, 256, 512, 1024]:
                configs.append(triton.Config({'BLOCK_SIZE_E': e, 'BLOCK_SIZE_N': n}, num_warps=w, num_stages=s))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['E','N']
    )

@triton.jit
def embedding_kernel(
    x_ptr,
    W_q_ptr,
    output_ptr,
    E, N,
    BLOCK_SIZE_E: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_e = pid_e * BLOCK_SIZE_E + tl.arange(0, BLOCK_SIZE_E)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_x = offs_e < E
    mask_w = (offs_e < E)[:, None] & (offs_n < N)[None, :]

    x = tl.load(x_ptr + offs_e, mask=mask_x, other=0)

    w_offset = x[:, None] * N + offs_n[None, :]
    out = tl.load(W_q_ptr + w_offset, mask=mask_w, other=0)

    tl.store(output_ptr + offs_e[:, None] * N + offs_n[None, :], out)

def embed_triton(x: torch.Tensor, W_q: torch.Tensor) -> torch.Tensor:
    M, L = x.shape
    K, N = W_q.shape 
    E = M * L

    output = torch.empty((M, L, N), dtype=W_q.dtype, device=W_q.device)

    grid = lambda META: (
        triton.cdiv(E, META['BLOCK_SIZE_E']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )

    embedding_kernel[grid](x, W_q, output, E, N)
    return output

###########################################################################################################
from triton.testing import do_bench
def eval_time(fct, params): 
    return do_bench(lambda: fct(**params), rep=1000) 

torch.manual_seed(0)
#embed = model.model.embed_tokens
embed = torch.nn.Embedding(128256, 4096, device='cuda', dtype=torch.float16)
W_q = embed.weight.data

#x = torch.randint(0, 100, (16, 2048), device='cuda', dtype=torch.int32)
x = torch.randint(0, 100, (64, 4096), device='cuda', dtype=torch.int32)

#Test correctness
out_ref = embed(x) 
out = embed_triton(x, W_q)
assert (out_ref - out).abs().mean().item() < 1e-5, "output mismatch"

#Test speed
ref = eval_time(lambda x: embed(x), {'x':x})
new = eval_time(embed_triton, {'x':x, 'W_q':W_q})
print('Speed-up',  '|', ref / new)


#A100 PCIE
#-------------------------------
#16 x 2048: 7.802747303866781
#64 x 4096: 8.558396433241947
