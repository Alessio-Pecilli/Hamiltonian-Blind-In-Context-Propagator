"""Block roll out with scheduled sampling: p=0 feeds back the true frames, p=1 the model's own."""
import torch

from model.propagator import block_rollout
from test_model import make_model, random_inputs


def test_block_rollout_feedback(tiny_cfg):
    tiny_cfg["model"]["residual"] = True
    model = make_model(tiny_cfg)
    node, edge, dt = (x.double() for x in random_inputs(2, 20, 4))
    T, anchors = torch.tensor([9, 7]), torch.tensor([[9], [7]])
    with torch.no_grad():
        for p in (0.0, 1.0):
            parts = list(block_rollout(model, node, edge, dt, anchors, T, 8, 4, p))
            assert len(parts) == 2 and parts[1][1].tolist() == [[13], [11]]
            n2, e2 = node.clone(), edge.clone()
            if p == 1.0:  # second block must be predicted from the first block's own frames
                for b, t in enumerate(T.tolist()):
                    n2[b, t + 1: t + 5] = parts[0][0]["node"][b, 0]
                    e2[b, t + 1: t + 5] = parts[0][0]["edge"][b, 0]
            ref = model(n2, e2, dt, T[:, None] + 4, T + 4, 4)
            assert torch.allclose(parts[1][0]["node"], ref["node"], atol=1e-10)
            assert torch.allclose(parts[1][0]["edge"], ref["edge"], atol=1e-10)
    assert torch.equal(node, random_inputs(2, 20, 4)[0].double())  # inputs untouched
