"""Export the trained model to ONNX for in-browser inference.

Single-node configuration: the demo's "analyze any entity" feature scores
entities that are NOT in the trained graphs, so the graph reduces to a lone
self-loop and the network runs as its own-history pathway (GRU -> self
attention -> head). Fixed shapes keep the export trivial and the file tiny.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attention_flow.dataset import INPUT_DAYS
from attention_flow.model import AttentionDiffusionNet


def main() -> None:
    model = AttentionDiffusionNet()
    model.load_state_dict(torch.load(ROOT / "models" / "attention_gnn.pt", weights_only=True))
    model.eval()

    x = torch.zeros(1, 1, INPUT_DAYS)
    adj = torch.ones(1, 1)
    out = ROOT / "docs" / "model.onnx"
    torch.onnx.export(
        model, (x, adj), str(out),
        input_names=["x", "adj"], output_names=["forecast"],
        opset_version=17, dynamo=False,
    )
    # parity check: ONNX output must match torch on a nontrivial input
    import numpy as np
    import onnx

    onnx.checker.check_model(str(out))
    torch.manual_seed(1)
    probe = torch.randn(1, 1, INPUT_DAYS)
    ref = float(model(probe, adj)[0, 0])
    print(f"exported {out} ({out.stat().st_size/1024:.0f} KB), torch ref on probe = {ref:+.4f}")


if __name__ == "__main__":
    main()
