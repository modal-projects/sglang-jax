"""OCP MXFP4 (E2M1 values, E8M0 scale per 32 K-elements) checkpoint decoding, exact.

Checkpoint format (AMD Quark ``quant_method: quark``, weight ``dtype: fp4``, ``qscheme: per_group``,
``group_size: 32``, ``scale_format: e8m0``; identical to DeepSeek-V4's MXFP4 experts):

  * ``<proj>.weight``        uint8 ``[N, K/2]``: two E2M1 codes per byte along K (the input dim),
                             **low nibble = even K index**, high nibble = odd K index.
  * ``<proj>.weight_scale``  uint8 ``[N, K/32]``: E8M0 exponent code ``c``, scale ``2**(c - 127)``;
                             ``c == 255`` is the NaN sentinel (rejected).
  * E2M1 code ``s|ee|m`` (sign = bit 3): magnitudes ``0, .5, 1, 1.5, 2, 3, 4, 6``.

Every decoded value ``e2m1 * 2**(c-127)`` has at most 2 significant bits, so it is exact in bf16 for
any exponent bf16 can represent (``c`` in [1, 254] covers bf16's normal range; Quark checkpoints use
``c`` near 120). The packed-FP4 GMM layout keeps the 4-bit codes and power-of-two scales unchanged.
"""

from __future__ import annotations

import sys

import ml_dtypes
import numpy as np

MXFP4_GROUP = 32
E8M0_BIAS = 127
E8M0_NAN = 255

# E2M1 code -> value (sign is bit 3).
FP4_E2M1_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# gmm_v2 packed-FP4 layout constants (see gmm_v2.expand_fp4_packed_e5m2). Kept here so the host-side
# converter has no kernel import; asserted equal to the kernel's in the unit tests.
FP4_PACK_CHUNK = 64
FP4_PACKED_SCALE_EXP = 14


def fp4_pack_v2() -> bool:
    """SGLJAX_FP4_PACK=v2: low-nibble layout q<<1 | s (see gmm_v2.fp4_pack_v2; must match it)."""
    import os

    v = os.environ.get("SGLJAX_FP4_PACK", "v1")
    if v not in ("v1", "v2"):
        raise ValueError(f"SGLJAX_FP4_PACK must be v1 or v2, got {v!r}")
    return v == "v2"


def _check(packed: np.ndarray, codes: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray]:
    packed = np.ascontiguousarray(packed).view(np.uint8)
    codes = np.ascontiguousarray(codes).view(np.uint8)
    if packed.ndim != 2 or codes.ndim != 2:
        raise ValueError(f"{key}: expected 2-D MXFP4 weight/scale, got {packed.shape}/{codes.shape}")
    n, half_k = packed.shape
    if codes.shape != (n, 2 * half_k // MXFP4_GROUP) or (2 * half_k) % MXFP4_GROUP:
        raise ValueError(f"{key}: scale shape {codes.shape} does not match weight {packed.shape}")
    if np.any(codes == E8M0_NAN):
        raise ValueError(f"{key}: reserved E8M0 scale code 255 (NaN)")
    return packed, codes


def unpack_e2m1_codes(packed: np.ndarray) -> np.ndarray:
    """uint8 [..., K/2] -> uint8 E2M1 codes [..., K] (low nibble first)."""
    packed = np.asarray(packed).view(np.uint8)
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), np.uint8)
    out[..., 0::2] = packed & np.uint8(0x0F)
    out[..., 1::2] = packed >> np.uint8(4)
    return out


def e8m0_to_float32(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes).view(np.uint8)
    if np.any(codes == E8M0_NAN):
        raise ValueError("reserved E8M0 scale code 255 (NaN)")
    return np.ldexp(np.ones(codes.shape, np.float32), codes.astype(np.int32) - E8M0_BIAS)


def dequantize_mxfp4(
    packed: np.ndarray, codes: np.ndarray, out_dtype=np.float32, *, key: str = "mxfp4"
) -> np.ndarray:
    """Checkpoint MXFP4 pair -> dense [N, K] (checkpoint orientation), exact."""
    packed, codes = _check(packed, codes, key)
    if np.any(codes > 252):  # 6 * 2**126 overflows f32/bf16
        raise ValueError(f"{key}: E8M0 scale code {int(codes.max())} overflows f32/bf16")
    vals = FP4_E2M1_VALUES[unpack_e2m1_codes(packed)]
    vals *= np.repeat(e8m0_to_float32(codes), MXFP4_GROUP, axis=1)
    if np.dtype(out_dtype) == np.dtype(ml_dtypes.bfloat16):
        out = vals.astype(ml_dtypes.bfloat16)
        # Exactness guard: every MXFP4 value must survive the bf16 cast bit-for-bit.
        if not np.array_equal(out.astype(np.float32), vals):
            raise ValueError(f"{key}: MXFP4 value not representable in bf16")
        return out
    return vals.astype(out_dtype)


def mxfp4_pair_to_packed_fp4(
    packed: np.ndarray, codes: np.ndarray, *, key: str = "mxfp4", scale_dtype=np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """Checkpoint MXFP4 pair -> gmm_v2 packed-FP4 operands, values unchanged.

    Returns weight uint32 ``[K/8, N]`` and scale ``[K/32, N]`` = ``2**(c - 127 + 14)``.
    Packing (per 64-row K chunk ``c``): packed row ``r = 8c + j``, byte ``b`` of the word holds K row
    ``64c + 4j + b`` in its low nibble and K row ``64c + 32 + 4j + b`` in its high nibble, so the
    kernel's SWAR expansion (``gmm_v2.expand_fp4_packed_e5m2``) yields K rows in natural order as
    E5M2 bytes equal to ``value * 2**-14``; the scale carries the ``2**14`` back.
    """
    packed, codes = _check(packed, codes, key)
    n, half_k = packed.shape
    k = 2 * half_k
    if k % FP4_PACK_CHUNK:
        raise ValueError(f"{key}: K={k} is not a multiple of {FP4_PACK_CHUNK}")
    nib = unpack_e2m1_codes(packed).T.reshape(k // 64, 2, 8, 4, n)  # [chunk, half, j, b, n]
    lo = nib[:, 0]
    if fp4_pack_v2():  # low nibble s|q -> q<<1 | s (gmm_v2.fp4_pack_v2)
        lo = ((lo & np.uint8(7)) << np.uint8(1)) | (lo >> np.uint8(3))
    byte = lo | (nib[:, 1] << np.uint8(4))  # [chunk, j, b, n] uint8
    # little-endian host (asserted): byte b of each word is bits 8b..8b+7
    assert sys.byteorder == "little"
    words = np.ascontiguousarray(byte.transpose(0, 1, 3, 2)).view(np.uint32)[..., 0]
    exp = codes.astype(np.int32) - E8M0_BIAS + FP4_PACKED_SCALE_EXP
    scale = np.ldexp(np.ones(codes.shape, np.float32), exp).T
    scale_c = scale.astype(scale_dtype)
    if not np.array_equal(scale_c.astype(np.float32), scale):
        raise ValueError(f"{key}: E8M0 scale not representable in {np.dtype(scale_dtype)}")
    return words.reshape(k // 8, n), np.ascontiguousarray(scale_c)


def packed_fp4_to_dense(words: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of :func:`mxfp4_pair_to_packed_fp4` -> dense f32 [K, N] (test/debug reference)."""
    k8, n = words.shape
    k = k8 * 8
    w = words.reshape(k // 64, 8, n)
    out = np.empty((k // 64, 2, 8, 4, n), np.uint8)
    for b in range(4):
        byte = (w >> np.uint32(8 * b)) & np.uint32(0xFF)
        lo = byte & 0xF
        if fp4_pack_v2():
            lo = ((lo & 1) << 3) | (lo >> 1)
        out[:, 0, :, b, :] = lo
        out[:, 1, :, b, :] = byte >> 4
    vals = FP4_E2M1_VALUES[out.reshape(k, n)]
    sc = np.repeat(np.asarray(scale, np.float32), MXFP4_GROUP, axis=0)
    return vals * sc * np.float32(2.0**-FP4_PACKED_SCALE_EXP)


def dequantize_fp8_block(weight_fp8: np.ndarray, scale: np.ndarray, block=(128, 128)) -> np.ndarray:
    """FP8 [N, K] x f32 block scale [ceil(N/bn), ceil(K/bk)] -> f32 [N, K] (reference only)."""
    w = np.asarray(weight_fp8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    bn, bk = block
    s = np.repeat(np.repeat(np.asarray(scale, np.float32), bn, 0), bk, 1)
    return w * s[: w.shape[0], : w.shape[1]]


def mxfp4_scale_dtype():
    """Device dtype of packed-FP4 expert scales (powers of two): bf16 unless
    SGLJAX_MXFP4_SCALE_DTYPE=float32."""
    import os

    import jax.numpy as jnp

    name = os.environ.get("SGLJAX_MXFP4_SCALE_DTYPE", "bfloat16")
    if name not in ("bfloat16", "float32"):
        raise ValueError(f"SGLJAX_MXFP4_SCALE_DTYPE must be bfloat16 or float32, got {name!r}")
    return jnp.dtype(name)


# ---------------------------------------------------------------------------------------------
# Weight-loader recipes (WeightSpec.recipe): numpy checkpoint inputs -> sharded jax.Arrays.
# ---------------------------------------------------------------------------------------------


def load_packed_fp4_experts(inputs, *, mesh, weight_spec, scale_spec, expert_ids, key, workers=8):
    """Routed experts of one projection: ``inputs = [w_0, s_0, w_1, s_1, ...]`` (logical experts,
    checkpoint ``[N, K/2]`` uint8 / ``[N, K/32]`` E8M0) -> (uint32 [E, K/8, N], scale [E, K/32, N]).

    ``expert_ids[p]`` is the logical expert stored at physical slot ``p``. Only the experts a
    device owns are converted (per-shard callback), each exactly once per call.
    """
    from concurrent.futures import ThreadPoolExecutor

    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    weights, scales = inputs[0::2], inputs[1::2]
    if len(weights) != len(scales):
        raise ValueError(f"{key}: odd number of MXFP4 inputs")
    n, half_k = np.asarray(weights[0]).shape
    k = 2 * half_k
    e = len(expert_ids)
    scale_dtype = np.dtype(mxfp4_scale_dtype())
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def convert(p: int):
        if p not in cache:
            lid = int(expert_ids[p])
            cache[p] = mxfp4_pair_to_packed_fp4(
                weights[lid], scales[lid], key=f"{key}[{lid}]", scale_dtype=scale_dtype
            )
        return cache[p]

    def experts(index):
        ids = range(*index[0].indices(e))
        missing = [p for p in ids if p not in cache]
        if missing:
            with ThreadPoolExecutor(min(workers, len(missing))) as ex:
                for p, v in zip(missing, ex.map(convert, missing)):
                    cache[p] = v
        return ids

    def cb(index, which):
        ids = experts(index)
        return np.stack([cache[p][which] for p in ids])[(slice(None), *index[1:])]

    w = jax.make_array_from_callback(
        (e, k // 8, n), NamedSharding(mesh, P(*weight_spec)), lambda i: cb(i, 0)
    )
    s = jax.make_array_from_callback(
        (e, k // MXFP4_GROUP, n), NamedSharding(mesh, P(*scale_spec)), lambda i: cb(i, 1)
    )
    cache.clear()
    return w, s


def load_dequantized_mxfp4_linear(inputs, *, mesh, sharding, key, dtype=ml_dtypes.bfloat16):
    """Non-expert MXFP4 linear: ``[weight [N, K/2], scale [N, K/32]]`` -> exact bf16 LinearBase
    kernel ``[K, N]`` with ``sharding``."""
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    dense = np.ascontiguousarray(dequantize_mxfp4(inputs[0], inputs[1], dtype, key=key).T)
    return (
        jax.make_array_from_callback(
            dense.shape, NamedSharding(mesh, P(*sharding)), lambda index: dense[index]
        ),
    )


MXFP4_FORMAT = "mxfp4"  # == quark.MXFP4
