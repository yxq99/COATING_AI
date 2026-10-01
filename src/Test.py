
import torch

# ❌ 危险写法
DTYPE = torch.double
BOUNDS_bad = torch.tensor([[0.5, 1, 0.2, 5, 30], [5, 8, 2, 60, 120]], dtype=DTYPE)
# print(BOUNDS_bad.dtype)  # torch.float32 (PyTorch 默认值)
# print(BOUNDS_bad.shape)
# print(BOUNDS_bad)


x=(BOUNDS_bad[0]>1).all(-1)
print(x)

print(BOUNDS_bad[0].all(-1))
print(BOUNDS_bad[1].sum(-1))