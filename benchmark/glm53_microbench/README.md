GLM-5.3 v6e routed-expert microbenchmarks (copies of the glm53-tpu/microbench harness so TPU jobs
can run them from a sglang-jax checkout; PYTHONPATH=<repo>/python:<this dir>).

- `mb_mxfp4_gemv.py`: one-chip EPMoE expert layer, packed MXFP4 vs fp8/bf16/int4, EP8/EP16,
  MXFP4 decode vs full-K paths and the cutoff summary (`--cpu-smoke`, `--aot`, or TPU timing).
- `mb_mxfp4_aot.py`: compile-only v6e checks of the MXFP4 GMMs and the EPMoE module on v6e-8.
- `mb_common.py`: shared harness (modes, AOT compile helper, recorder, timing).
