"""CPU unit tests for the opt-in fused decode switch and SGLANG_STEP_TIMING."""

import logging
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.scheduler import StepTimingAccumulator
from sgl_jax.srt.managers.tp_worker import fused_decode_sample_enabled


def _args(**kw):
    base = dict(
        aot_model_dir=None,
        pd_disaggregation="",
        speculative_algorithm=None,
        disaggregation_mode="null",
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SGLANG_FUSE_DECODE_SAMPLE", raising=False)
    monkeypatch.delenv("SGLANG_PD_FUSE_SAMPLE", raising=False)


def test_fused_decode_default_off():
    assert not fused_decode_sample_enabled(_args())


def test_fused_decode_opt_in(monkeypatch):
    monkeypatch.setenv("SGLANG_FUSE_DECODE_SAMPLE", "1")
    assert fused_decode_sample_enabled(_args())
    assert not fused_decode_sample_enabled(_args(), is_draft_worker=True)
    assert not fused_decode_sample_enabled(_args(speculative_algorithm="DFLASH"))
    assert not fused_decode_sample_enabled(_args(aot_model_dir="/x"))
    assert not fused_decode_sample_enabled(_args(disaggregation_mode="decode"))


def test_pathways_switch_unchanged(monkeypatch):
    a = _args(pd_disaggregation="pathways")
    assert not fused_decode_sample_enabled(a)
    monkeypatch.setenv("SGLANG_FUSE_DECODE_SAMPLE", "1")
    assert not fused_decode_sample_enabled(a)  # pathways keeps its own switch
    monkeypatch.setenv("SGLANG_PD_FUSE_SAMPLE", "1")
    assert fused_decode_sample_enabled(a)


def test_step_timing_logs_every_n(caplog):
    mode = SimpleNamespace(is_decode=lambda: True)
    batch = SimpleNamespace(batch_size=lambda: 4, forward_mode=mode)
    acc = StepTimingAccumulator(node_rank=2, every=3)
    with caplog.at_level(logging.INFO, logger="sgl_jax.srt.managers.scheduler"):
        t = 0.0
        for _ in range(3):
            acc.add(t, t + 0.001, t + 0.0011, t + 0.0012, t + 0.002, t + 0.004,
                    (0.0001, 0.0015, 0.0), batch)
            t += 0.0045
    msgs = [r.getMessage() for r in caplog.records if "[step-timing r2]" in r.getMessage()]
    assert len(msgs) == 1
    assert "n=3 decode=3 bs=4.0" in msgs[0]
    assert "recv=1.000" in msgs[0] and "d2h=1.500" in msgs[0]
    assert acc.n == 0  # window reset
