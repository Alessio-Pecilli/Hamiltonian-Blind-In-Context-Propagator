import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.generate import generate_file  # noqa: E402


@pytest.fixture(scope="session")
def tiny_h5(tmp_path_factory):
    """Small TFIM dataset (n=6) shared by the tests."""
    p = tmp_path_factory.mktemp("data") / "tiny.h5"
    generate_file(str(p), n=6, ntraj=8, L=60, dt=0.1, family="tfim", seed=7, verbose=False)
    return str(p)


@pytest.fixture(scope="session")
def tiny_cfg():
    from config import load_config
    cfg = load_config(os.path.join(ROOT, "configs/config.yaml"), "local")
    cfg["model"].update(d=32, de=8, nh=4, L=2, m=8, K=8)
    cfg["train"].update(H=6, T_min=8, T_max=24, anchors=2)
    return cfg
