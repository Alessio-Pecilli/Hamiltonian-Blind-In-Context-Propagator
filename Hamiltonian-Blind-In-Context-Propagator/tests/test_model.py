"""Steps 3-6 of the implementation plan: equivariance, causality, stability, loss, overfit."""
import copy

import torch

from data.dataset import collate, gather_targets, TrajectoryDataset
from losses import total_loss
from model.head_generator import GeneratorHead
from model.propagator import Propagator


def random_inputs(B, T, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    node = torch.rand(B, T, n, 3, generator=g) * 2 - 1
    e = torch.rand(B, T, n, n, 9, generator=g) * 2 - 1
    swap = [3 * b + a for a in range(3) for b in range(3)]
    e = 0.5 * (e + e.transpose(2, 3)[..., swap])
    e = e * (~torch.eye(n, dtype=torch.bool))[None, None, :, :, None]
    return node, e, torch.full((B,), 0.1)


def make_model(cfg, head="A"):
    cfg = copy.deepcopy(cfg)
    cfg["model"]["head"] = head
    cfg["model"]["bf16"] = False
    torch.manual_seed(0)
    m = Propagator(cfg).double().eval()
    if head == "A":  # W_A is zero-initialised: perturb so that the coupling path is tested
        torch.nn.init.normal_(m.head.w_A.weight, std=0.05)
        if m.head.p:  # zero-initialised too: perturb so that the pair latent path is tested
            torch.nn.init.normal_(m.head.w_C.weight, std=0.3)
            torch.nn.init.normal_(m.head.D_u.weight, std=0.3)
    return m


def test_permutation_equivariance(tiny_cfg):
    for head in ("A", "B"):
        model = make_model(tiny_cfg, head)
        B, T, n = 2, 12, 6
        node, edge, dt = random_inputs(B, T, n)
        node, edge, dt = node.double(), edge.double(), dt.double()
        anchors, Tl = torch.tensor([[11, 5], [11, 9]]), torch.tensor([11, 11])
        perm = torch.randperm(n)
        with torch.no_grad():
            out = model(node, edge, dt, anchors, Tl)
            outp = model(node[:, :, perm], edge[:, :, perm][:, :, :, perm], dt, anchors, Tl)
        assert torch.allclose(outp["node"], out["node"][..., perm, :], atol=1e-5)
        assert torch.allclose(outp["edge"], out["edge"][..., perm, :, :][..., perm, :], atol=1e-5)
        if "probe" in out:
            assert torch.allclose(outp["probe"]["J"], out["probe"]["J"][:, perm][:, :, perm], atol=1e-5)


def test_causality(tiny_cfg):
    """Outputs at anchor t must not depend on frames > t."""
    model = make_model(tiny_cfg)
    node, edge, dt = random_inputs(1, 16, 5)
    node, edge, dt = node.double(), edge.double(), dt.double()
    anchors, T = torch.tensor([[7]]), torch.tensor([15])
    node2, edge2 = node.clone(), edge.clone()
    node2[:, 8:] = 0.3
    edge2[:, 8:] *= -1
    with torch.no_grad():
        a = model(node, edge, dt, anchors, T)["node"]
        b = model(node2, edge2, dt, anchors, T)["node"]
    assert torch.allclose(a, b, atol=1e-10)


def test_same_code_path_for_different_n(tiny_cfg):
    model = make_model(tiny_cfg)
    for n in (6, 8, 10):
        node, edge, dt = random_inputs(2, 10, n)
        out = model(node.double(), edge.double(), dt.double(), torch.tensor([[9], [9]]),
                    torch.tensor([9, 9]))
        assert out["node"].shape == (2, 1, tiny_cfg["train"]["H"], n, 3)


def test_generator_stability():
    torch.manual_seed(1)
    head = GeneratorHead(d=16, de=8, m=6, K=5, basis_std=0.5).double()
    torch.nn.init.normal_(head.w_A.weight, std=1.0)
    h, e = torch.randn(4, 7, 16).double() * 3, torch.randn(4, 7, 7, 8).double() * 3
    G, *_ = head.generator(h, e)
    assert torch.linalg.eigvals(G).real.max() <= 1e-5
    P = torch.linalg.matrix_exp(0.1 * G)
    Z = torch.randn(4, 42).double()
    z0 = Z.norm(dim=-1)
    for _ in range(50):
        Z = (P @ Z[..., None]).squeeze(-1)
        assert (Z.norm(dim=-1) <= z0 + 1e-9).all()


def test_loss_permutation_invariant(tiny_cfg, tiny_h5):
    model = make_model(tiny_cfg).float()
    ds = TrajectoryDataset(tiny_h5, (12, 12), tiny_cfg["train"]["H"], anchors=2, permute=False)
    torch.manual_seed(0)
    batch = collate([ds[0], ds[1]])
    perm = torch.randperm(ds.n)
    pb = dict(batch)
    pb["node"], pb["edge"] = batch["node"][:, :, perm], batch["edge"][:, :, perm][:, :, :, perm]
    pb["adj"], pb["J"] = batch["adj"][:, perm][:, :, perm], batch["J"][:, perm][:, :, perm]
    pb["h"] = batch["h"][:, perm]
    losses = []
    for b in (batch, pb):
        tn, te = gather_targets(b, tiny_cfg["train"]["H"])
        with torch.no_grad():
            pred = model(b["node"], b["edge"], b["dt"], b["anchors"], b["T"])
            losses.append(total_loss(pred, b, tn, te, tiny_cfg)[0])
    assert torch.allclose(losses[0], losses[1], rtol=1e-5)


def test_single_batch_overfit_decreases(tiny_cfg, tiny_h5):
    """Short version of the Stage-1 overfit check (the full one is `train.py --overfit`)."""
    torch.manual_seed(0)
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"]["bf16"] = False
    model = Propagator(cfg)
    ds = TrajectoryDataset(tiny_h5, (16, 16), cfg["train"]["H"], anchors=1, permute=False)
    batch = collate([ds[0]])
    tn, te = gather_targets(batch, cfg["train"]["H"])
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    first = None
    for _ in range(150):
        pred = model(batch["node"], batch["edge"], batch["dt"], batch["anchors"], batch["T"])
        loss, logs = total_loss(pred, batch, tn, te, cfg)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or logs["pred"]
    assert logs["pred"] < 0.2 * first


def test_edge_decoder_symmetric_with_asymmetric_beta():
    """Eq. (23): E[j,i,(b,a)] = E[i,j,(a,b)] must hold for any beta, not only beta = 0."""
    torch.manual_seed(0)
    head = GeneratorHead(d=16, de=8, m=6, K=5).double()
    with torch.no_grad():
        head.beta.copy_(torch.randn(9).double())
    z = torch.randn(3, 2, 5, 6).double()
    _, edge = head.decode(z)
    swap = [3 * b + a for a in range(3) for b in range(3)]
    assert torch.allclose(edge, edge.transpose(-2, -3)[..., swap], atol=1e-12)


def test_generator_without_dissipation_is_skew():
    """use_S=False: G = A - A^T exactly, S_F = 0, the latent norm is conserved."""
    torch.manual_seed(2)
    head = GeneratorHead(d=16, de=8, m=6, K=5, basis_std=0.5, use_S=False).double()
    torch.nn.init.normal_(head.w_A.weight, std=1.0)
    h, e = torch.randn(3, 5, 16).double(), torch.randn(3, 5, 5, 8).double()
    G, _, _, S_F = head.generator(h, e)
    assert torch.allclose(G, -G.transpose(-1, -2), atol=1e-12)
    assert (S_F == 0).all()
    P = torch.linalg.matrix_exp(0.1 * G)
    Z = torch.randn(3, 30).double()
    assert torch.allclose((P @ Z[..., None]).squeeze(-1).norm(dim=-1), Z.norm(dim=-1), atol=1e-9)


def test_new_pair_components_are_active(tiny_cfg):
    """When switched on, the three architecture options change the output."""
    tiny_cfg = copy.deepcopy(tiny_cfg)
    tiny_cfg["model"].update(pair_attn=True, triangle=True, pair_latent=8)
    node, edge, dt = random_inputs(1, 12, 5)
    node, edge, dt = node.double(), edge.double(), dt.double()
    args = (node, edge, dt, torch.tensor([[11]]), torch.tensor([11]))
    out = {}
    for name, off in (("all", {}), ("no_pair_attn", {"pair_attn": False}), ("no_triangle", {"triangle": False}),
                      ("no_latent", {"pair_latent": 0})):
        cfg = copy.deepcopy(tiny_cfg)
        cfg["model"].update(off)
        torch.manual_seed(0)
        m = Propagator(cfg).double().eval()
        torch.nn.init.normal_(m.head.w_A.weight, std=0.05)
        if m.head.p:
            torch.nn.init.normal_(m.head.w_C.weight, std=0.3); torch.nn.init.normal_(m.head.D_u.weight, std=0.3)
        with torch.no_grad():
            out[name] = m(*args)["edge"]
    for k in ("no_pair_attn", "no_triangle", "no_latent"):
        assert not torch.allclose(out["all"], out[k], atol=1e-6), k


def test_options_off_gives_the_old_parameter_set(tiny_cfg):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"].update(pair_attn=False, triangle=False, pair_latent=0)
    keys = Propagator(cfg).state_dict().keys()
    assert not any(k for k in keys if ".pa_" in k or ".tri_" in k or "head.w_u" in k or "head.w_C" in k or "head.D_u" in k)


def test_pair_latent_keeps_edge_symmetry_and_is_stable():
    torch.manual_seed(3)
    head = GeneratorHead(d=16, de=8, m=6, K=5, pair_latent=4).double()
    for w in (head.w_C.weight, head.D_u.weight):
        torch.nn.init.normal_(w, std=1.0)
    h = torch.randn(2, 5, 16).double()
    e = torch.randn(2, 5, 5, 8).double()
    e = 0.5 * (e + e.transpose(1, 2))
    out = head(h, e, torch.full((2,), 0.1).double(), horizon=7)
    swap = [3 * b + a for a in range(3) for b in range(3)]
    assert torch.allclose(out["edge"], out["edge"].transpose(-2, -3)[..., swap], atol=1e-12)
    u = head.pair_rollout(e, torch.full((2,), 0.1).double(), 30)
    assert (u.norm(dim=-1) <= u[:, 0:1].norm(dim=-1) * (1 + 1e-9) + 1e-9).all()  # skew generator: norm kept


def test_defaults_have_no_probe_and_no_probe_loss(tiny_cfg, tiny_h5):
    assert tiny_cfg["model"]["probe"] is False and tiny_cfg["loss"]["lambda_probe"] == 0
    assert not any(k.startswith("probe.") for k in Propagator(tiny_cfg).state_dict())
    old = copy.deepcopy(tiny_cfg)
    del old["model"]["probe"]  # old configs: probe present
    assert any(k.startswith("probe.") for k in Propagator(old).state_dict())
    ds = TrajectoryDataset(tiny_h5, (12, 12), tiny_cfg["train"]["H"], anchors=1, permute=False)
    b = collate([ds[0]])
    tn, te = gather_targets(b, tiny_cfg["train"]["H"])
    pred = Propagator(tiny_cfg)(b["node"], b["edge"], b["dt"], b["anchors"], b["T"])
    assert "probe" not in pred
    assert total_loss(pred, b, tn, te, tiny_cfg)[1]["probe"] == 0


def pauli_model(tiny_cfg, **kw):
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"].update(pauli_A=True, m=8, **kw)
    torch.manual_seed(0)
    m = Propagator(cfg).double().eval()
    torch.nn.init.normal_(m.head.coef[-1].weight, std=0.1)
    return m


def test_pauli_basis_real_orthogonal():
    from model.head_generator import pauli_basis
    B = pauli_basis(8, 2).double()
    assert len(B) == 1 + 3 * 3 + 9 * 3  # weight <= 2 on 3 qubits
    gram = torch.einsum("sab,tab->st", B, B)
    assert torch.allclose(gram, 8 * torch.eye(len(B)).double(), atol=1e-12)
    assert len(pauli_basis(8, 0)) == 64
    try:
        pauli_basis(6)
        assert False
    except ValueError:
        pass


def test_pauli_A_skew_equivariant_causal(tiny_cfg):
    model = pauli_model(tiny_cfg)
    node, edge, dt = random_inputs(2, 12, 5)
    node, edge, dt = node.double(), edge.double(), dt.double()
    anchors, Tl = torch.tensor([[11], [11]]), torch.tensor([11, 11])
    perm = torch.randperm(5)
    with torch.no_grad():
        out = model(node, edge, dt, anchors, Tl)
        outp = model(node[:, :, perm], edge[:, :, perm][:, :, :, perm], dt, anchors, Tl)
        assert torch.allclose(out["G"], -out["G"].transpose(-1, -2), atol=1e-12)
        assert out["gate"].max() <= 1 + 1e-9 and out["diag_norm"].shape == (2, 1, 5)
        assert torch.allclose(outp["node"], out["node"][..., perm, :], atol=1e-5)
        assert torch.allclose(outp["gate"], out["gate"][..., perm, :][..., perm], atol=1e-5)
        node2, edge2 = node.clone(), edge.clone()
        node2[:, 8:] = 0.3
        edge2[:, 8:] *= -1
        a = model(node, edge, dt, torch.tensor([[7], [7]]), Tl)["node"]
        b = model(node2, edge2, dt, torch.tensor([[7], [7]]), Tl)["node"]
        assert torch.allclose(a, b, atol=1e-10)


def test_quadratic_readout(tiny_cfg):
    import numpy as np
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"].update(readout="quadratic", use_S=False)
    model = make_model(cfg)
    node, edge, dt = random_inputs(2, 12, 5)
    node, edge, dt = node.double(), edge.double(), dt.double()
    anchors, T = torch.tensor([[11], [11]]), torch.tensor([11, 11])
    with torch.no_grad():
        out = model(node, edge, dt, anchors, T)
        assert out["node"].shape == (2, 1, cfg["train"]["H"], 5, 3)
        assert torch.allclose(out["edge"], out["edge"].transpose(3, 4)[..., [3 * b + a for a in range(3) for b in range(3)]], atol=1e-10)
        # the anchor latent is renormalised: its scale must not matter
        model.head.w_z.weight.mul_(7.0); model.head.w_z.bias.mul_(7.0)
        out2 = model(node, edge, dt, anchors, T)
    assert torch.allclose(out["node"], out2["node"], atol=1e-8)
    # mechanism: rotations at w1=3 and w2=2 give a node signal at w1+-w2 (and none at 3)
    head = GeneratorHead(d=8, de=4, m=4, K=2, use_S=False, readout="quadratic").double().eval()
    with torch.no_grad():
        head.C.zero_(); head.C[0, 0, 2] = 1.0
        t = torch.arange(64) * 2 * np.pi / 64
        z = torch.stack([torch.cos(3 * t), torch.sin(3 * t), torch.cos(2 * t), torch.sin(2 * t)], -1)
        node_t, _ = head.decode(z.double()[:, None, :])
    spec = np.abs(np.fft.rfft(node_t[:, 0, 0].numpy()))
    assert spec[5] > 10 * spec[3] and spec[1] > 10 * spec[3] and spec[5] > 1 and spec[1] > 1
