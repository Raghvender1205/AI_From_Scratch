import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    A simple feed-forward net that serves as an "expert"
    """
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )
    
    def forward(self, x):
        return self.net(x)
    
class MoE(nn.Module):
    """
    A Mixture-of-Experts (MoE) layer. This layer routes each token to a 
    subset of experts (top-k) using a gating network and combines their 
    output with a weighted sum 
    """
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([Expert(d_model, d_model * 2) for _ in range(n_experts)])
        self.gating_network = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor):
        """
        Args: 
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # (bs * seq_len, d_model)

        # Gating weights
        gating_logits = self.gating_network(x_flat) # (..., n_experts)
        top_k_weights, top_k_indices = torch.topk(gating_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1) # (bs * seq_len, top_k)
        
        # Output tensor
        final_output = torch.zeros_like(x_flat)

        # NOTE: Not optimized, just for explaining
        for i in range(self.n_experts):
            # Find which tokens in the batch are routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                # Pass relevant tokens through expert
                expert_inputs = x_flat[expert_mask]
                expert_outputs = self.experts[i](expert_inputs)

                # Find where to place the outputs
                # Get idx in the flattended tensor that correspond to this expert
                for k in range(self.top_k):
                    token_indices = torch.where((top_k_indices[:, k] == i) & expert_mask)[0]
                    if token_indices.numel() > 0:
                        # Get corresponding weights
                        weights = top_k_weights[token_indices, k].unsqueeze(-1) 
                        # get corresponding expert outputs
                        expert_outputs_for_tokens = expert_outputs[:len(token_indices)]
                        # Add weighted output to the final result
                        final_output[token_indices] += weights * expert_outputs_for_tokens

        return final_output.view(batch_size, seq_len, d_model)


class TransformerBlock(nn.Module):
    """
    A standard Transformer block with multi-head self attention
    and MoE as the feed forward net.
    """
    def __init__(self, d_model, n_heads, n_experts, top_k=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe = MoE(d_model, n_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # Attn sub-layer with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # MoE sublayer with residual connection and layer norm
        moe_output = self.moe(x)
        x = self.norm2(x + moe_output)

        return x

class LLM(nn.Module):
    """
    A minimal Large Language Model (LLM) using Transformer block with MoE
    """
    def __init__(self, vocab_size, d_model, n_heads, n_experts, n_layers, top_k=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_experts, top_k)
            for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Args: 
            x (torch.Tensor): Input tensor of token IDs, Shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output logits, Shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.output_layer(x)

        return x
    

if __name__ == '__main__':
    # Constants
    BATCH_SIZE = 8
    SEQ_LEN = 32

    # Model Parameters
    vocab_size = 10_000 
    d_model = 256 # model dimension
    n_heads = 4
    n_experts = 8 # no. of experts in MoE layer
    n_layers = 2   # no. of transformer blocks
    top_k = 2   # no. of experts to route each token to

    # Model
    model = LLM(
        vocab_size=vocab_size,
        d_model=d_model, 
        n_heads=n_heads,
        n_experts=n_experts,
        n_layers=n_layers,
        top_k=top_k
    )

    # Dummy tensor
    dummy_input = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN))

    # Get model output
    output = model(dummy_input)

    print(f"Input Shape (token IDs): {dummy_input.shape}")
    print(f"Output shape (logits): {output.shape}")

    assert output.shape == (BATCH_SIZE, SEQ_LEN, vocab_size)
