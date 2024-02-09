import torch
from hedgehog.main import Hedgehog

# Creat tokens
x = torch.randint(0, 100, (1, 100))

# Create model
model = Hedgehog(
    num_tokens=100,
    dim=512,
    head_dim=512,
)

# Forward
out = model(x)


# Out
print(out)
