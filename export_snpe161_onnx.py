"""
export_onnx_snpe161.py -- export BowShockTCN to ONNX in a form SNPE 1.61 converts correctly.

Why this exists
---------------
SNPE 1.61 has no real Conv1d. It rebuilds each Conv1d as a Conv2d and, when the
Conv1d has padding built in, it copies that padding onto the extra height axis
too -- which broke all 10 causal convs. This script hands SNPE 1.61 explicit
Conv2d layers (kernel 1 x k, padding on the time axis only), so that faulty
translation step never runs.

What it does NOT change
-----------------------
tcn_model.py and your trained weights. The swap happens only on a copy of the
model inside this script, and the output is checked against the original
PyTorch model before the file is saved.

Run on the server (venv), then convert in the 1.61 container:
  snpe-onnx-to-dlc --input_network best_model_snpe161.onnx --output_path best_model_snpe161.dlc
"""
import copy
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from tcn_model import BowShockTCN

CHECKPOINT = "checkpoints/best_model.pt"            # EDIT: your best-validation checkpoint (same one export_onnx.py uses)
OUTPUT = "best_model_snpe161.onnx"
TIME_STEPS, ENERGY_BINS = 128, 63
OPSET = 11                              # same opset as your original export; well supported by SNPE 1.x


class Conv1dAsConv2d(nn.Module):
    """Same math as a padded nn.Conv1d, written as a Conv2d with a height of 1."""

    def __init__(self, conv1d: nn.Conv1d):
        super().__init__()
        k, d, p = conv1d.kernel_size[0], conv1d.dilation[0], conv1d.padding[0]
        self.conv = nn.Conv2d(conv1d.in_channels, conv1d.out_channels,
                              kernel_size=(1, k), dilation=(1, d), padding=(0, p))   # height pad 0, time pad p
        with torch.no_grad():
            self.conv.weight.copy_(conv1d.weight.unsqueeze(2))   # (out, in, k) -> (out, in, 1, k)
            self.conv.bias.copy_(conv1d.bias)

    def forward(self, x):                                        # x: (batch, channels, time)
        return self.conv(x.unsqueeze(2)).squeeze(2)


class LinearAsConv1x1(nn.Module):
    """Same math as the final nn.Linear, written as a 1x1 conv over time.
    The classic exporter writes a Linear on 3D input as MatMul; your original
    export had a 1x1 Conv here, which SNPE 1.61 already converted correctly."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.conv = nn.Conv1d(linear.in_features, linear.out_features, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.copy_(linear.weight.unsqueeze(2))   # (out, in) -> (out, in, 1)
            self.conv.bias.copy_(linear.bias)

    def forward(self, x):                                        # x: (batch, time, channels)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


# 1. Load the trained model exactly as training saved it
model = BowShockTCN()
state = torch.load(CHECKPOINT, map_location="cpu")
if isinstance(state, dict) and "model_state_dict" in state:     # some checkpoints wrap the weights
    state = state["model_state_dict"]
model.load_state_dict(state)
model.eval()

# 2. Make an export copy and swap each padded Conv1d (block.0 and block.4 in every TemporalBlock)
#    and the output Linear, using only layer types SNPE 1.61 handled correctly in your first DLC
export_model = copy.deepcopy(model)
swapped = 0
for block in export_model.tcn:
    for idx in (0, 4):
        if isinstance(block.block[idx], nn.Conv1d) and block.block[idx].padding[0] > 0:
            block.block[idx] = Conv1dAsConv2d(block.block[idx])
            swapped += 1
export_model.output_layer[1] = LinearAsConv1x1(export_model.output_layer[1])
export_model.eval()
print(f"Swapped {swapped} padded Conv1d layers for Conv2d equivalents (+ output Linear -> 1x1 conv)")

# 3. Export with a fixed input shape and the weights inside the one .onnx file
dummy = torch.randn(1, TIME_STEPS, ENERGY_BINS)
torch.onnx.export(export_model, dummy, OUTPUT,
                  input_names=["input"], output_names=["logits"],
                  opset_version=OPSET, do_constant_folding=True,
                  dynamo=False)            # classic exporter: one self-contained file, no .onnx.data
onnx.checker.check_model(onnx.load(OUTPUT))

# 4. Verify: the new ONNX must match the ORIGINAL PyTorch model
sess = ort.InferenceSession(OUTPUT)
worst = 0.0
with torch.no_grad():
    for seed in range(5):
        x = torch.randn(1, TIME_STEPS, ENERGY_BINS, generator=torch.Generator().manual_seed(seed))
        diff = np.abs(sess.run(None, {"input": x.numpy()})[0] - model(x).numpy()).max()
        worst = max(worst, float(diff))
print(f"Max |ONNX - original PyTorch| over 5 inputs: {worst:.2e}  (expect ~1e-5 or smaller)")

# 5. Confirm no padded Conv1d is left for SNPE 1.61 to mistranslate
graph = onnx.load(OUTPUT).graph
bad = [n.name for n in graph.node if n.op_type == "Conv"
       and len(next(a.ints for a in n.attribute if a.name == "kernel_shape")) == 1
       and any(next((a.ints for a in n.attribute if a.name == "pads"), []))]
print("Padded Conv1d layers remaining:", len(bad), "(expect 0)")
print(f"Saved {OUTPUT}")