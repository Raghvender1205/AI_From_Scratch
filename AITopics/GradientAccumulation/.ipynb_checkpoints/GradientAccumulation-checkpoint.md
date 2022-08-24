{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d072d2e2",
   "metadata": {},
   "source": [
    "## Gradient Accumulation\n",
    "\n",
    "When we train a Neural Net, we usually divide our data into mini-batches to prevent `MemoryLimitError` or `ResourceExhaustError` (Out Of Memory) and make the data fit on GPU. We then go through these mini-batches one by one. The Neural net then `predicts` labels which are then used to compute `loss` w.r.t to the actual targets. Then, we perform `backward` pass to compute `gradients` and `update` model weights in the direction of those gradients.\n",
    "\n",
    "`Gradient Accumulation` modifies the last step of the `Training` as instead of updating the network weights on every `batch`, we save the `gradient` values and proceed to the next `batch` and add up the new `gradients`. The weight update is done only after the batches have been processed by the model. So, it basically `imitates` a `large batch size`. Imagine we want to use 32 images in one batch but the hardware does'nt allow to go beyond 8. So, we use 8 batches and update weights once every `4` batches. If we accumulate `gradients` from every batch in between, result will (almost) be the same and we will be able to perform training\n",
    "\n",
    "## How it Works\n",
    "This is what a standard `training` loop looks like"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2c23c18",
   "metadata": {},
   "source": [
    "```python\n",
    "import torch\n",
    "\n",
    "for (inputs, labels) in dataloader:\n",
    "    # Extract labels and inputs\n",
    "    inputs = inputs.to(device)\n",
    "    labels = labels.to(device)\n",
    "    \n",
    "    with torch.set_grad_enabled(True):\n",
    "        # Forward pass\n",
    "        preds = model(inputs)\n",
    "        loss = criterion(preds, labels)\n",
    "        \n",
    "        # Backward pass\n",
    "        loss.backward()\n",
    "        \n",
    "        # Update Weights\n",
    "        optimizer.step()\n",
    "        optimizer.zero_grad()\n",
    "```\n",
    "\n",
    "Now, we implement `gradient accumulation` on this\n",
    "1. Specify the `accum_iter` parameter. This indicates once in how many batches we will update weights\n",
    "2. Condition the weight update on the idx of the running batch. The requires using `enumerate(dataloader)` to store batch_idx when iterating through data\n",
    "3. Divide the running `loss` by `accum_iter`. This normalizes loss to reduce the contribution of each `mini-batch` we are processing. \n",
    "\n",
    "Depending on the way to compute `loss` we may not need the last step. If we `average loss` within each batch, the division is then already correct and then there is no need for extra `normalization`.\n",
    "\n",
    "```python\n",
    "accum_iter = 4\n",
    "\n",
    "for batch_idx, (inputs, labels) in enumerate(dataloader):\n",
    "    # Extract labels and inputs\n",
    "    inputs = inputs.to(device)\n",
    "    labels = labels.to(device)\n",
    "    \n",
    "    with torch.set_grad_enabled(True):\n",
    "        # Forward pass\n",
    "        preds = model(inputs)\n",
    "        loss = criterion(preds, labels)\n",
    "        \n",
    "        # Normalize loss to account for batch accumulation\n",
    "        loss = loss / accum_iter\n",
    "        \n",
    "        # Backward pass\n",
    "        loss.backward()\n",
    "        \n",
    "        # Update Weights\n",
    "        if ((batch_idx+1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):\n",
    "            optimizer.step()\n",
    "            optimizer.zero_grad()\n",
    "```"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
