# Mixture of Experts (MoE)

A `Mixture-of-Experts` model is a neural architecture that consists of multiple smaller `sub-networks` called <b>`experts`</b>, alongside a <b>`gating`</b> network. The gate dynamically routes each input to only a few relevant experts rather than using the entire model, enabling `conditional computation`

A minimal implementation is written in `moe.py`.

## 1. Experts
They are independently trained `sub-networks`. Each expert learns to handle a specific type of input pattern or task. For example, in LLMs on expert might specialize in financial text while another in casual conversation

## 2. Gating Network (Router)
A module that decides which experts should process a given input. 

It assigns a score or probability to each expert based on the input and then selects the top-k experts with the highest scores.

There are some routing strategies
* Top-K selection: Choose top-k scoring experts 
* Softmax gating: Compute the weighted combination of expert outputs.
* Load-balanced gating: Include auxiliary losses to balance expert usage.


## 3. Sparse Activation / Conditional Computation
Instead of activating all experts for every input, only a small subset is activated , typically `top-1` or `top-2`

This keeps the computation efficient even as the number of experts scales up. In this way

* Activates only ∼10% or fewer experts per input
* Saves computation and memory
* Allows large-scale model with reasonable inference

## 4. Aggregation Layer
It merges the outputs of the selected experts to form the final output for the user input.

It uses out of the following methods below

* Weighted sum based on the gating scores.
* If weighting is used, then simple average.
* Concatenation or attention

It ensures that even though only a few experts are used, their outputs integrate seamlessly with the rest of the model

## 5. Load Balancing (Auxiliary Loss)
A mechanism to prevent the router from always picking the same few experts.

During training, auxiliary loss term is added
- Encourage equal usage of all experts
- Avoid expert underutilization or overfitting

An example of load balancing maybe `route regularization` in [Switch Transformer](https://arxiv.org/abs/2101.03961)

## 6. Expert Capacitly Limit
Now, sometimes too many tokens get routed to one expert which can lead to memory overflow, batch size issues.

So, in this we cap the number of tokens each expert can handle per step (eg. 128 tokens/expert) and use `overflow` strategies like 

* Random dropping
* Re-routing to next-best expert.