"""Verify pinned Qwen3-8B weight files before starting the server."""

import hashlib
import json
from pathlib import Path


FILES = {
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}


if __name__ == "__main__":
    model_dir = Path("w2/models/Qwen3-8B")
    result = {"model": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "files": []}
    for name, expected in FILES.items():
        digest = hashlib.sha256()
        with (model_dir / name).open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        actual = digest.hexdigest()
        assert actual == expected, (name, actual, expected)
        result["files"].append({"file": name, "bytes": (model_dir / name).stat().st_size, "sha256": actual})
    config = json.loads((model_dir / "config.json").read_text())
    assert (config["num_hidden_layers"], config["num_key_value_heads"], config["head_dim"]) == (36, 8, 128)
    result["model_config"] = config
    Path("w2/results/model_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
