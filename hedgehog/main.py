import torch
from torch import nn, Tensor
from zeta.nn import FeedForward

def softmax_attn(
    q: Tensor,
    k: Tensor,
):
    scale = q.shape[-1] ** -0.5
    qk = torch.einsum("bhmd, bhnd -> bhmn", q, k) * scale
    return torch.softmax(qk, dim=-1)


def quadratic_linear_attn(
    q: Tensor,
    k: Tensor,
):
    qk = torch.einsum("bhnd, bhnd -> bhmn", q, k)
    return qk / qk.sum(dim=-1, keepdim=True)



class HedgeHogModule(nn.Module):
    """
    HedgeHogModule is a PyTorch module that applies linear transformation
    followed by an activation function to the input tensor.

    Args:
        head_dim (int): The dimension of the input tensor.
        activation (str, optional): The activation function to be applied.
            Defaults to "exp".

    Attributes:
        head_dim (int): The dimension of the input tensor.
        activation (str): The activation function to be applied.
        layer (nn.Linear): The linear transformation layer.

    Methods:
        init_weights: Initializes the weights of the linear layer.
        forward: Performs forward pass through the module.

    """

    def __init__(
        self,
        dim: int,
        activation: str = "exp",
    ):
        super().__init__()
        self.dim = dim
        self.activation = activation
        self.layer = nn.Linear(dim, dim)
        self.init_weights_()

    def init_weights_(self):
        nn.init.eye_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass through the module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying linear transformation
            and activation function.

        """
        x = self.layer(x)  # Shape BATCH, HEADS, SEQLEN, DIMENSION
        return torch.cat([torch.exp(x), torch.exp(-x)], dim=1)



class HedgeHogAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        training: bool = True,
        output_attentions: bool = False,
        qk_norm: bool = False,
        *args,
        **kwargs,
    ):
        """
        HedgeHogAttention module that performs attention computation.

        Args:
            dim (int): The input dimension of the module.
            base_attn: The base attention module.
            training (bool, optional): Whether the module is in training mode. Defaults to True.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.training = training
        self.qk_norm = qk_norm
        self.output_attentions = output_attentions

        # Trainable maps
        self.mlp_q = HedgeHogModule(dim)
        self.mlp_k = HedgeHogModule(dim)
        self.mlp_v = HedgeHogModule(dim)

        # Freeze params
        if not self.training:
            for p in self.base_attn.parameters():
                p.requires_grad = False

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # If qk norm
        if qk_norm:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the HedgeHogAttention module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The concatenated tensor of q, k, and v.
        """
        # Compute maps
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if self.qk_norm:
            q, k = self.norm(q), self.norm(k)

        # Apply the mlp
        q = self.mlp_q(q)
        k = self.mlp_k(k)
        v = self.mlp_v(v)

        concat = q + k + v
        print(f"concat shape: {concat.shape}")

        return concat


class Hedgehog(nn.Module):
    """
    Hedgehog module for performing attention-based computations.

    Args:
        num_tokens (int): Number of tokens in the input.
        dim (int): Dimension of the input.
        heads (int, optional): Number of attention heads. Defaults to 8.
        depth (int, optional): Number of layers. Defaults to 4.
        head_dim (int, optional): Dimension of each attention head. Defaults to 64.
        mult (int, optional): Multiplier for the feedforward layer. Defaults to 4.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Attributes:
        dim (int): Dimension of the input.
        heads (int): Number of attention heads.
        depth (int): Number of layers.
        head_dim (int): Dimension of each attention head.
        mult (int): Multiplier for the feedforward layer.
        dropout (float): Dropout probability.
        layers (nn.ModuleList): List of attention and feedforward layers.
        emb (nn.Embedding): Embedding layer.
        norm (nn.LayerNorm): Layer normalization.
        to_out (nn.Sequential): Sequential layer for output transformation.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        heads: int = 8,
        depth: int = 4,
        head_dim: int = 64,
        mult: int = 4,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.head_dim = head_dim
        self.mult = mult
        self.dropout = dropout

        # layers
        self.layers = nn.ModuleList([])

        # Embedding
        self.emb = nn.Embedding(num_tokens, dim)

        # Add both the attention and the feedforward
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        HedgeHogAttention(
                            dim=dim,
                            head_dim=head_dim,
                        ),
                        FeedForward(
                            dim=dim,
                            mult=mult,
                            dropout=dropout,
                            *args,
                            **kwargs,
                        ),
                    ]
                )
            )

        # norm
        self.norm = nn.LayerNorm(dim)

        # To out
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Softmax(dim=-1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        print(f"x embedding shape: {x.shape}")
        for attn, ff in self.layers:
            x = attn(x) + x
            print(f"x attn shape: {x.shape}")
            x = ff(x) + x
        return self.to_out(x)
