from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.util import fit_block


def test_fit_block_prefers_divisor_over_padding():
    assert fit_block(4096, 1024, 256) == 1024  # already divides
    assert fit_block(1536, 1024, 256) == 768  # GLM-5.3 q_a / dense gate_up per chip
    assert fit_block(1536, 256, 128) == 256
    assert fit_block(576, 1024, 256) == 1024  # no aligned divisor -> keep (pad)
    assert fit_block(128, 1024, 256) == 1024  # smaller than one tile -> keep
