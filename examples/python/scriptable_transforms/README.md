# Scriptable transformations examples

## Inference example

`inference.py` shows how to combine input's transformation and model's prediction and use `torch.jit.script` to obtain
a single scripted module.

```bash
python inference.py
```