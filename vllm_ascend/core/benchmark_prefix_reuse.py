# SPDX-License-Identifier: Apache-2.0
"""Single-prefix HBM copy/reuse for a dedicated benchmark server only.

Not a production prefix cache. Native PC must be disabled. All restored pages
and recurrent state are private to the consumer; snapshots are immutable.
"""
from __future__ import annotations

import json
import math
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

ENABLED = os.environ.get("VLLM_ASCEND_BENCHMARK_PREFIX_REUSE") == "1"
_worker = None


def read_marker():
    directory = os.environ["VLLM_ASCEND_ATTENTION_TIMING_DIR"]
    path = os.environ.get("VLLM_ASCEND_ATTENTION_TIMING_MARKER",
                          str(Path(directory) / "active.json"))
    return json.loads(Path(path).read_text())


def validate_config(config):
    if (not config.model_config.enforce_eager or
            config.model_config.hf_config.model_type != "deepseek_v4" or
            config.parallel_config.data_parallel_size != 1 or
            config.parallel_config.pipeline_parallel_size != 1 or
            config.scheduler_config.max_num_seqs != 1 or
            config.scheduler_config.max_num_batched_tokens not in (128, 256) or
            config.scheduler_config.async_scheduling or
            config.speculative_config is not None or
            config.kv_transfer_config is not None or
            config.cache_config.enable_prefix_caching or
            config.cache_config.block_size != 128):
        raise ValueError("Benchmark prefix reuse requires deepseek_v4, eager, "
                         "DP=PP=1, max-num-seqs=1, block-size=128, chunk=128/256, "
                         "no async/MTP/connector/native prefix caching")
    if not os.environ.get("VLLM_ASCEND_ATTENTION_TIMING_DIR"):
        raise ValueError("Benchmark prefix reuse requires attention timing directory")


def marker_key(marker):
    return (marker["case"], int(marker["repeat"]))


class PrefixController:
    """Scheduler-side exact-token match and successful-prime publication."""
    def __init__(self):
        self.ready = None

    def admit(self, marker, tokens):
        if marker.get("cache_mode", "benchmark") != "benchmark":
            raise ValueError("Server benchmark reuse mode disagrees with driver")
        phase = marker["phase"]
        if phase not in ("warmup", "prefill", "measure"):
            raise ValueError("Invalid benchmark phase")
        prefix = int(marker["prefix_tokens"])
        total = int(marker["input_tokens"])
        if total % 128 or prefix < 0 or prefix >= total or (prefix and (prefix < 384 or prefix % 128)):
            raise ValueError("Prefix must be >=384, aligned to 128, and shorter than input")
        expected = prefix if phase == "prefill" else total
        if len(tokens) != expected or not tokens:
            raise ValueError("Benchmark prompt token count mismatch")
        plan = dict(marker, tokens=tuple(tokens), cached_tokens=0)
        if phase in ("warmup", "prefill"):
            self.ready = None
        elif prefix:
            if self.ready is None or marker_key(self.ready) != marker_key(marker):
                raise ValueError("No ready benchmark prefix for this case/repeat")
            if tuple(tokens[:prefix]) != self.ready["tokens"]:
                raise ValueError("Benchmark prefix tokens differ from successful prime")
            plan["cached_tokens"] = prefix
            self.ready = None  # Single-use: accidental replay must not become a hit.
        else:
            self.ready = None
        return plan

    def finish(self, plan, success):
        if plan["phase"] == "prefill":
            self.ready = plan if success else None


def page_count(prefix, ratio, block_size):
    return ((prefix // ratio) + block_size - 1) // block_size


class SnapshotStore:
    def __init__(self, quota_bytes, row_view=None):
        self.quota = quota_bytes
        self.row_view = row_view or (lambda tensor: tensor)
        self.entries = {}
        self.bytes = 0

    def save(self, layer, kv, block_ids, state, state_ids, prefix):
        kv = tuple(self.row_view(t) for t in kv)
        state = tuple(self.row_view(t) for t in state)
        if layer in self.entries:
            raise ValueError("Duplicate prefix snapshot for layer")
        size = sum(len(indices) * math.prod(t.shape[1:]) * t.element_size()
                   for tensors, indices in ((kv, block_ids), (state, state_ids))
                   for t in tensors)
        if self.bytes + size > self.quota:
            raise ValueError("Benchmark HBM snapshot quota exceeded")
        # Advanced indexing copies WHOLE physical rows, including partial tails.
        # No aliases to allocator-owned source blocks are retained.
        saved_kv = tuple(t[block_ids] for t in kv)
        saved_state = tuple(t[state_ids] for t in state)
        self.entries[layer] = (prefix, saved_kv, saved_state)
        self.bytes += size
        return size

    def restore(self, layer, kv, block_ids, state, state_ids, prefix):
        kv = tuple(self.row_view(t) for t in kv)
        state = tuple(self.row_view(t) for t in state)
        if layer not in self.entries:
            raise ValueError("Missing layer prefix snapshot")
        saved_prefix, saved_kv, saved_state = self.entries[layer]
        if (saved_prefix != prefix or len(kv) != len(saved_kv) or
                len(state) != len(saved_state)):
            raise ValueError("Incompatible prefix snapshot")
        for targets, sources, indices in ((kv, saved_kv, block_ids),
                                          (state, saved_state, state_ids)):
            for target, source in zip(targets, sources):
                if tuple(source.shape) != (len(indices), *target.shape[1:]):
                    raise ValueError("Prefix snapshot shape mismatch")
                target[indices] = source
        return sum(t.numel() * t.element_size() for t in (*saved_kv, *saved_state))


class WorkerSnapshots:
    def __init__(self, npu, rank, quota, row_view=None):
        self.npu, self.rank, self.quota = npu, rank, quota
        self.row_view = row_view
        self.key = None
        self.store = SnapshotStore(quota, row_view)
        self.restored = set()

    def select(self, marker):
        key = marker_key(marker)
        if self.key != key:
            self.store = SnapshotStore(self.quota, self.row_view)
            self.restored.clear()
            self.key = key

    def copy(self, action, marker, layer, kv, blocks, state, state_ids, prefix,
             wall_started=None):
        begin_wall = time.perf_counter() if wall_started is None else wall_started
        start = self.npu.Event(enable_timing=True)
        end = self.npu.Event(enable_timing=True)
        start.record()
        if action == "save":
            size = self.store.save(layer, kv, blocks, state, state_ids, prefix)
        else:
            if layer in self.restored:
                raise ValueError("Duplicate prefix restore; request replay/preemption is unsupported")
            size = self.store.restore(layer, kv, blocks, state, state_ids, prefix)
            self.restored.add(layer)
        end.record()
        end.synchronize()
        row = dict(case=marker["case"], repeat=marker["repeat"], phase=marker["phase"],
                   action=action, layer=layer, rank=self.rank, prefix_tokens=prefix,
                   bytes=size, device_ms=start.elapsed_time(end),
                   wall_ms=(time.perf_counter()-begin_wall)*1000,
                   pid=os.getpid(), host=socket.gethostname())
        directory = Path(os.environ["VLLM_ASCEND_ATTENTION_TIMING_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"prefix-{row['host']}-rank{self.rank}-pid{os.getpid()}.jsonl"
        with (directory / filename).open("a") as stream:
            stream.write(json.dumps(row) + "\n")


@contextmanager
def prefix_scope(layer, metadata, kv, state, ratio, config):
    global _worker
    if not ENABLED or metadata is None:
        yield
        return
    validate_config(config)
    marker = read_marker()
    if _worker is None:
        import torch
        import torch.distributed as dist
        from vllm_ascend.ascend_config import get_ascend_config
        ascend = get_ascend_config()
        if ascend.enable_kv_tnd or ascend.multistream_dsa_preprocess:
            raise ValueError("Benchmark reuse requires PA KV layout and single-stream DSA")
        _worker = WorkerSnapshots(torch.npu, dist.get_rank() if dist.is_initialized() else 0,
                                  int(os.environ.get("VLLM_ASCEND_BENCHMARK_SNAPSHOT_BYTES",
                                                     str(2 * 1024**3))),
                                  # Preserve FP8 payload bits without requiring FP8
                                  # gather/scatter implementations in torch-npu.
                                  row_view=lambda tensor: tensor.view(torch.uint8))
    _worker.select(marker)
    prefix = int(marker["prefix_tokens"])
    if marker["phase"] == "warmup" or not prefix:
        yield
        return
    if metadata.num_prefills != 1 or metadata.num_decodes != 0:
        raise ValueError("Benchmark reuse only supports one prefill request")
    lookup_started = time.perf_counter()
    end_position = int(metadata.seq_lens[0].item())
    context = end_position - int(metadata.num_actual_tokens)
    restore = marker["phase"] == "measure" and context == prefix
    save = marker["phase"] == "prefill" and end_position == prefix
    if marker["phase"] == "measure" and context < prefix:
        raise ValueError("Scheduler did not skip benchmark prefix")
    if marker["phase"] == "measure" and context > prefix and layer not in _worker.restored:
        raise ValueError("Suffix started without restoring its prefix snapshot")
    if restore or save:
        # Ring rotation changes table ordering, not physical offsets within state.
        ring = metadata.prefill.state_block_table[0].tolist()
        state_ids = sorted(set(int(x) for x in ring if x != 0))
        if len(state_ids) != 3 or state_ids != list(range(state_ids[0], state_ids[0]+3)):
            raise ValueError("Expected complete contiguous three-block state ring")
        if ratio == 1:
            kv, blocks = (), []
        else:
            count = page_count(prefix, ratio, 128)
            blocks = [int(x) for x in metadata.prefill.block_table[0, :count].tolist()]
            if len(blocks) != count or len(set(blocks)) != count or any(x <= 0 for x in blocks):
                raise ValueError("Incomplete benchmark compressed KV allocation")
        if restore:
            _worker.copy("restore", marker, layer, kv, blocks, state, state_ids, prefix,
                         wall_started=lookup_started)
    yield
    if save:
        _worker.copy("save", marker, layer, kv, blocks, state, state_ids, prefix)
