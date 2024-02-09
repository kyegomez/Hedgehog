import torch 
from torch import nn, Tensor
from einops import rearrange


def quadratic_linear_attn(
    q: Tensor, 
    k: Tensor,
):
    qk = torch.einsum(
        "bhnd, bhnd -> bhmn", q, k
    )
    return qk / qk.sum(dim=-1, keepdim=True)


class HedgeHogModule(nn.Module):
    def __init__(
        self,
        head_dim: int,
        activation: str = "exp",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.activation = activation
        self.layer = nn.Linear(head_dim, head_dim)
        self.init_weights()
    
    def init_weights_(self):
        nn.init.eye_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.layer(x) # Shape BATCH, HEADS, SEQLEN, DIMENSION
        return torch.cat(
            [torch.exp(x), torch.exp(-x)],
            dim=-1
        ),
        
        
class HedgeHogAttention(nn.Module):
    def __init__(
        self,
        base_attn,
        training: bool = True,
        output_attentions: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.base_attn = base_attn
        self.training = training
        self.output_attentions = output_attentions
        
        # Trainable maps
        self.mlp_q = HedgeHogModule(base_attn.head_dim)
        self.mlp_k = HedgeHogModule(base_attn.head_dim)
        
        # Freeze params
        for p in self.base_attn.parameters():
            p.requires_grad = False
        
        self.q_proj = self.base_attn.q_proj
        self.k_proj = self.base_attn.k_proj
        
    def forward(self, x: Tensor) -> Tensor:
        q, k, v = x
        
        # Compute maps
        q = self.mlp_q(
            self.q_proj(x)
        )
        
        k = self.mlp_k(
            self.k_proj(x)
        )
        
        # Pred attns 
        pred_attns = quadratic_linear_attn(q, k)
        
        
        # Output
        true_attns = self.base_attn(x)
        
        if self.output_attentions:
            return pred_attns, true_attns



class HedgehogBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        mult: int = 4,
        dropout: float = 0.1,
        *args,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.weight = nn.Parameter(torch.randn(heads, dim_head, dim_head))
        self.beta = nn.Parameter(torch.randn(heads, dim_head, dim_head))
        
        self.theta = torch.exp(self.weight.transpose(1, 2) + self.weight)
        
        