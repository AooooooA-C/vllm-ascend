# SPDX-License-Identifier: Apache-2.0
"""Opt-in, serialized eager-mode DeepSeek V4 attention diagnostics.

The marker is a local/shared file owned by the benchmark driver. No marker
means no measurement. Never enable this on a server serving other traffic.
"""
from __future__ import annotations

import contextvars
import json
import os
import socket
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

ENABLED = bool(os.environ.get("VLLM_ASCEND_ATTENTION_TIMING_DIR"))
_active = contextvars.ContextVar("ascend_attention_measurement", default=None)
_timer = None


class AttentionTimer:
    def __init__(self, directory, marker, npu, rank):
        self.directory = Path(directory)
        self.marker = Path(marker)
        self.npu = npu
        self.rank = rank
        self.sequence = 0

    def event(self):
        event = self.npu.Event(enable_timing=True)
        event.record()
        return event

    @contextmanager
    def module(self, layer, metadata):
        if metadata is None:
            yield
            return
        try:
            case = json.loads(self.marker.read_text())
        except FileNotFoundError:
            yield
            return
        if case.get("phase") not in ("warmup", "prefill", "measure"):
            raise ValueError("Invalid attention benchmark marker phase")
        if metadata.num_prefills + metadata.num_decodes != 1:
            raise ValueError("Attention timing requires exactly one request")
        # Read metadata before starting events; GPU->CPU copies must not be
        # included in attention timing. The diagnostic intentionally synchronizes.
        seq_lens = metadata.seq_lens.tolist()
        tokens = int(metadata.num_actual_tokens)
        record = dict(case=case["case"], phase=case["phase"],
                      repeat=int(case["repeat"]), layer=layer,
                      rank=self.rank, pid=os.getpid(), host=socket.gethostname(),
                      sequence=self.sequence, query_tokens=tokens,
                      context_tokens=int(seq_lens[0]) - tokens,
                      stage="prefill" if metadata.num_prefills else "decode")
        self.sequence += 1
        cores = []
        token = _active.set((self, cores))
        try:
            start = self.event()
            yield
            end = self.event()
            end.synchronize()
            record["module_ms"] = start.elapsed_time(end)
            record["core_ms"] = sum(a.elapsed_time(b) for _, a, b in cores)
            record["core_calls"] = len(cores)
            record["time_ns"] = time.time_ns()
            self.directory.mkdir(parents=True, exist_ok=True)
            filename = (f"attention-{record['host']}-rank{self.rank}-"
                        f"pid{os.getpid()}.jsonl")
            with (self.directory / filename).open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        finally:
            _active.reset(token)


@contextmanager
def core_scope(stage):
    active = _active.get()
    if active is None:
        yield
        return
    timer, cores = active
    start = timer.event()
    yield
    cores.append((stage, start, timer.event()))


def module_scope(layer, metadata, config):
    global _timer
    if not ENABLED or metadata is None:
        return nullcontext()
    if _timer is None:
        import torch
        import torch.distributed as dist
        if not config.model_config.enforce_eager:
            raise ValueError("Attention timing requires --enforce-eager")
        if config.parallel_config.data_parallel_size != 1:
            raise ValueError("Attention timing requires data_parallel_size=1")
        if config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Attention timing requires pipeline_parallel_size=1")
        if config.speculative_config is not None:
            raise ValueError("Attention timing requires speculative decoding off")
        directory = os.environ["VLLM_ASCEND_ATTENTION_TIMING_DIR"]
        marker = os.environ.get("VLLM_ASCEND_ATTENTION_TIMING_MARKER",
                                str(Path(directory) / "active.json"))
        rank = dist.get_rank() if dist.is_initialized() else 0
        _timer = AttentionTimer(directory, marker, torch.npu, rank)
    return _timer.module(layer, metadata)
