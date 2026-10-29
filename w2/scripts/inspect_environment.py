import importlib.metadata
import hashlib
import json
import os
import platform
from pathlib import Path

import sglang
import torch


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    properties = torch.cuda.get_device_properties(0)
    result = {
        "python": platform.python_version(), "sglang_source": sglang.__file__,
        "sglang_source_version": sglang.__version__,
        "packages": {name: version(name) for name in ["sglang", "torch", "transformers", "sgl-kernel", "flashinfer-python", "triton"]},
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name, "gpu_total_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "model": "Qwen/Qwen3-8B", "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "model_config": json.loads(Path("w2/models/Qwen3-8B/config.json").read_text()),
        "author_commit": "7ef897eaedc16626a281ccfe0d453fae80251e8f",
        "source_files_sha256": {
            name: hashlib.sha256(Path("w2/vendor/KVFlow/python/sglang/srt", name).read_bytes()).hexdigest()
            for name in ["managers/scheduler.py", "mem_cache/radix_cache.py"]
        },
        "image_digest": "sha256:fce5585fa8da175224ead70727a4660f719e34adf0638aebf6dd27930f03e1b6",
    }
    Path("w2/results/environment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
