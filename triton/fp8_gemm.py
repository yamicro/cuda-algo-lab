import torch
import triton
import triton.language as tl
import time

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offset, y)
    tl.store(s_ptr + pid, s)


BLOCK_SIZE = 1024
x = torch.randn(1 << 20, dtype=torch.float16, device='cuda')
y = torch.empty_like(x)
s = torch.empty((x.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,
                dtype=torch.float32, device='cuda')

grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
start_time = time.time()
act_quant_kernel[grid](
    x, y, s,
    BLOCK_SIZE=BLOCK_SIZE
)
end_time = time.time()
print("time usage is:", end_time - start_time)
print("y result is:", y[0: 5])
