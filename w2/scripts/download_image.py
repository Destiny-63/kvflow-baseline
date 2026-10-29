"""Download a pinned public image with aria2 and build a Docker load archive."""

import concurrent.futures
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


REPOSITORY = "lmsysorg/sglang"
DIGEST = "sha256:fce5585fa8da175224ead70727a4660f719e34adf0638aebf6dd27930f03e1b6"
TAG = "lmsysorg/sglang:v0.5.2-cu126"
ROOT = Path("w2/cache/image")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, url):
        return None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    registry = os.environ.get("W2_REGISTRY", "https://registry-1.docker.io").rstrip("/")
    auth = os.environ.get("W2_REGISTRY_AUTH", "https://auth.docker.io/token")
    service = os.environ.get("W2_REGISTRY_SERVICE", "registry.docker.io")
    token_cache = {"time": 0}
    token_lock = threading.Lock()

    def registry_headers():
        with token_lock:
            if time.monotonic() - token_cache["time"] > 180:
                query = urlencode({"service": service, "scope": f"repository:{REPOSITORY}:pull"})
                with urlopen(f"{auth}?{query}", timeout=30) as response:
                    payload = json.load(response)
                    token_cache["token"] = payload.get("token") or payload["access_token"]
                token_cache["time"] = time.monotonic()
            return {"Authorization": f"Bearer {token_cache['token']}", "Accept": ", ".join([
                "application/vnd.docker.distribution.manifest.v2+json",
                "application/vnd.oci.image.manifest.v1+json",
            ])}

    request = Request(f"{registry}/v2/{REPOSITORY}/manifests/{DIGEST}", headers=registry_headers())
    with urlopen(request, timeout=30) as response:
        raw = response.read()
    assert hashlib.sha256(raw).hexdigest() == DIGEST.split(":")[1]
    manifest = json.loads(raw)
    (ROOT / "registry_manifest.json").write_bytes(raw)
    descriptors = [manifest["config"], *manifest["layers"]]
    print(f"Pinned manifest verified; {len(manifest['layers'])} layers, {sum(d['size'] for d in descriptors) / 1e9:.2f} GB compressed", flush=True)

    def download(descriptor):
        expected = descriptor["digest"].split(":")[1]
        path = ROOT / expected
        if path.exists() and path.stat().st_size == descriptor["size"] and sha256(path) == expected:
            return path
        headers = registry_headers()
        request = Request(f"{registry}/v2/{REPOSITORY}/blobs/{descriptor['digest']}", headers=headers)
        try:
            with build_opener(NoRedirect()).open(request, timeout=30) as response:
                url = response.geturl()
                if descriptor["size"] < 1_000_000:
                    path.write_bytes(response.read())
                    assert sha256(path) == expected
                    print(f"Verified {expected[:12]} ({descriptor['size'] / 1e6:.1f} MB)", flush=True)
                    return path
        except HTTPError as error:
            if error.code not in [301, 302, 303, 307, 308]:
                print(f"Blob {expected[:12]} returned HTTP {error.code}", flush=True)
                raise
            url = error.headers["Location"]
        redirected = urlsplit(url).netloc != urlsplit(registry).netloc
        if descriptor["size"] < 1_000_000:
            request = Request(url, headers={} if redirected else headers)
            with urlopen(request, timeout=30) as response, path.open("wb") as output:
                shutil.copyfileobj(response, output)
        else:
            auth_args = [] if redirected else [f"--header=Authorization: {headers['Authorization']}"]
            with (ROOT / f"{expected}.log").open("w") as log:
                subprocess.run([
                    os.environ["W2_ARIA2"], "--continue=true", "--split=16",
                    "--max-connection-per-server=16", "--min-split-size=1M",
                    "--file-allocation=none", "--auto-file-renaming=false",
                    "--allow-overwrite=true", "--summary-interval=30",
                    "--show-console-readout=false", "--enable-color=false",
                    "--max-tries=5", "--retry-wait=5", "--connect-timeout=20",
                    f"--dir={ROOT}", f"--out={expected}", *auth_args, url,
                ], stdout=log, stderr=subprocess.STDOUT, check=True)
        assert path.stat().st_size == descriptor["size"] and sha256(path) == expected, descriptor["digest"]
        print(f"Verified {expected[:12]} ({descriptor['size'] / 1e6:.1f} MB)", flush=True)
        return path

    if os.environ.get("W2_METADATA_ONLY") == "1":
        for descriptor in descriptors:
            if descriptor["size"] < 1_000_000:
                download(descriptor)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(download, descriptors))
    config_digest = manifest["config"]["digest"].split(":")[1]
    config = json.loads((ROOT / config_digest).read_text())
    assert config["architecture"] == "amd64" and config["os"] == "linux"
    layers = []
    for descriptor, diff_id in zip(manifest["layers"], config["rootfs"]["diff_ids"], strict=True):
        name = descriptor["digest"].split(":")[1]
        layer = ROOT / f"{name}.tar"
        if not layer.exists():
            with gzip.open(ROOT / name, "rb") as source, layer.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        assert sha256(layer) == diff_id.split(":")[1], name
        layers.append(layer.name)
    docker_manifest = [{"Config": f"{config_digest}.json", "RepoTags": [TAG], "Layers": layers}]
    (ROOT / "manifest.json").write_text(json.dumps(docker_manifest))
    shutil.copyfile(ROOT / config_digest, ROOT / f"{config_digest}.json")
    archive = ROOT / "sglang-pinned.tar"
    with tarfile.open(archive, "w") as output:
        for name in ["manifest.json", f"{config_digest}.json", *layers]:
            output.add(ROOT / name, arcname=name)
    print(f"Docker archive ready: {archive}", flush=True)


if __name__ == "__main__":
    main()
