"""Record HiCache state changes without synchronizing CUDA or changing policy."""

import functools
import json
import os
import time


def emit(kind, **fields):
    path = os.environ.get("W2_CACHE_EVENTS")
    if not path:
        return
    record = {"event": kind, "time_ns": time.time_ns(), "pid": os.getpid(), **fields}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (json.dumps(record) + "\n").encode())
    finally:
        os.close(fd)


def install():
    import torch
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(HiRadixCache, "_w2_instrumented", False):
        return
    HiRadixCache._w2_instrumented = True

    def wrap_node_method(name, kind):
        original = getattr(HiRadixCache, name)

        @functools.wraps(original)
        def wrapped(self, node, *args, **kwargs):
            node_id = node.id
            result = original(self, node, *args, **kwargs)
            tokens = len(result) if name == "load_back" and result is not None else result
            if tokens is not None and tokens > 0:
                emit(kind, node_id=node_id, tokens=int(tokens))
            return result

        setattr(HiRadixCache, name, wrapped)

    wrap_node_method("write_backup", "backup_enqueued")
    wrap_node_method("_evict_backuped", "gpu_evicted_host_retained")
    wrap_node_method("load_back", "reload_enqueued")

    original_backup = MHATokenToKVPoolHost.backup_from_device_all_layer

    @functools.wraps(original_backup)
    def backup(self, device_pool, host_indices, device_indices, io_backend):
        result = original_backup(self, device_pool, host_indices, device_indices, io_backend)
        emit("backup_kernel_submitted", tokens=len(host_indices), backend=io_backend)
        return result

    MHATokenToKVPoolHost.backup_from_device_all_layer = backup
    original_load = MHATokenToKVPoolHost.load_to_device_per_layer

    @functools.wraps(original_load)
    def load(self, device_pool, host_indices, device_indices, layer_id, io_backend):
        result = original_load(self, device_pool, host_indices, device_indices, layer_id, io_backend)
        emit("reload_layer_submitted", tokens=len(host_indices), layer=layer_id, backend=io_backend)
        return result

    MHATokenToKVPoolHost.load_to_device_per_layer = load
    original_forward = ModelRunner.forward_extend
    pending = []

    @functools.wraps(original_forward)
    def forward(self, forward_batch, *args, **kwargs):
        for measurement in pending[:]:
            if measurement[1].query():
                start, end, start_ns, tokens = measurement
                emit("prefill_forward_completed", forward_start_ns=start_ns,
                     tokens=tokens, cuda_ms=start.elapsed_time(end))
                pending.remove(measurement)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_ns = time.time_ns()
        start.record()
        result = original_forward(self, forward_batch, *args, **kwargs)
        end.record()
        pending.append((start, end, start_ns, forward_batch.input_ids.numel()))
        return result

    ModelRunner.forward_extend = forward
