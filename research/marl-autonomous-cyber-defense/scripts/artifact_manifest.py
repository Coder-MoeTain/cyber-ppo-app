"""Create and verify SHA-256 manifests for research experiment artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

try:
    import torch
except ImportError:
    torch = None


def hardware_metadata():
    if torch is None:
        return {"torch_version": None, "cuda_version": None, "cuda_available": False, "gpus": []}
    available = bool(torch.cuda.is_available())
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": available,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if available else [],
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(paths, output_path):
    cwd = Path.cwd().resolve()
    output_resolved = output_path.resolve()
    found = set()
    for raw in paths:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {raw}")
        candidates = [path] if path.is_file() else [
            item for item in path.rglob("*") if item.is_file()
        ]
        for item in candidates:
            resolved = item.resolve()
            if resolved == output_resolved:
                continue
            try:
                relative = resolved.relative_to(cwd)
            except ValueError as exc:
                raise ValueError(f"Artifact must be inside the current workspace: {item}") from exc
            found.add(relative.as_posix())
    if not found:
        raise ValueError("No artifact files found")
    return sorted(found)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_manifest(paths, output):
    files = []
    for relative in collect_files(paths, output):
        path = Path(relative)
        files.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hardware": hardware_metadata(),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported or malformed artifact manifest")
    failures = []
    for entry in manifest["files"]:
        item = Path(entry["path"])
        if not item.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        actual_size = item.stat().st_size
        actual_hash = sha256_file(item)
        if actual_size != entry["size_bytes"] or actual_hash != entry["sha256"]:
            failures.append(f"modified: {entry['path']}")
    if failures:
        raise ValueError("Artifact verification failed: " + "; ".join(failures))
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--paths", nargs="+", required=True)
    create.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if args.command == "create":
        manifest = create_manifest(args.paths, Path(args.output))
        print(json.dumps({
            "manifest": args.output,
            "files": len(manifest["files"]),
            "git_commit": manifest["git_commit"],
        }, indent=2))
    else:
        count = verify_manifest(Path(args.manifest))
        print(json.dumps({"manifest": args.manifest, "verified_files": count}, indent=2))


if __name__ == "__main__":
    main()
