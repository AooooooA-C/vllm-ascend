# SPDX-License-Identifier: Apache-2.0
# Adapted from local origin/main (4f49ae332), attention/sparse_flash_mla.py.
"""Minimal A5 BF16 paged MLA adapter for eager text-only benchmarks."""
from functools import lru_cache
from importlib import import_module


@lru_cache()
def load_ops():
    try:
        import_module("cann_ops_transformer")
        import torch
        ns = torch.ops.cann_ops_transformer
        return ns.sparse_flash_mla, ns.sparse_flash_mla_metadata
    except (ImportError, AttributeError) as error:
        raise RuntimeError("A5 BF16 benchmark requires matching CANN 9.2 "
                           "cann_ops_transformer SparseFlashMla and metadata operators") from error


def adapt(kwargs, metadata=False):
    kwargs = dict(kwargs)
    kwargs["layout_kv"] = "PA_BBND"
    kwargs.pop("device", None)
    kwargs.pop("cu_seqlens_ori_kv", None)
    kwargs.pop("cu_seqlens_cmp_kv", None)
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = int(kwargs.pop("max_seqlen_kv"))
    if metadata and "max_seqlen_q" in kwargs:
        kwargs["max_seqlen_q"] = int(kwargs["max_seqlen_q"])
    ratio = kwargs.get("cmp_ratio", 1)
    lengths = kwargs.get("seqused_ori_kv")
    if ratio > 1 and lengths is not None:
        kwargs["seqused_cmp_kv"] = lengths // ratio
        kwargs["cmp_residual_kv"] = lengths % ratio
        if metadata and "max_seqlen_ori_kv" in kwargs:
            kwargs["max_seqlen_cmp_kv"] = kwargs["max_seqlen_ori_kv"] // ratio
    return kwargs


def metadata_op(**kwargs):
    _, op = load_ops()
    return op(**adapt(kwargs, metadata=True))


def attention_op(q, **kwargs):
    from vllm_ascend.attention.benchmark_timer import core_scope
    if kwargs.get("sinks") is None:
        raise ValueError("SparseFlashMla requires the layer's float32 sinks")
    op, _ = load_ops()
    kwargs = adapt(kwargs)
    # Tensor length preparation belongs to module time, not the core op span.
    with core_scope("sparse_flash_mla"):
        return op(q, **kwargs)
