#!/usr/bin/env python3
"""M4.3-REQ-005.4 — OCI archive/layer scanner (Design.md §7.4).

Exports the image with `docker save`, then walks every layer tar entry
(never `extractall()` — only `getmembers()`, so the scanner itself never
writes attacker-controlled paths to disk) looking for forbidden content and
path-traversal member names. Whiteout markers (`.wh.*`, OCI opaque
`.wh..wh..opq`) are not content and are skipped, but the file they
shadowed still lives, byte for byte, in whichever earlier layer wrote it —
OCI layers are additive, so this scanner intentionally inspects every
layer independently rather than a squashed final view. A secret added in
layer N and `rm`'d (whiteout) in layer N+1 is still a member of layer N and
is still classified there (hosted-CI remediation iteration 1 regression
test: test_deleted_credential_still_detected_in_earlier_layer).

Hosted-CI remediation iteration 1 (see
docs/milestones/m4.3-artifact-deployment-safety/Hosted_CI_Remediation_Iteration_1.md)
narrowed the generic `.pem` credential pattern below with a fail-closed CA
allowlist: legitimate OS/interpreter trust-store bundles (Debian
`/etc/ssl/certs`, `/usr/lib/ssl`, `/usr/share/ca-certificates`, RHEL
`/etc/pki/...`, and any vendored `certifi/cacert.pem`) are exempted only
when BOTH the path matches a known trust-store location AND — for regular
files, which is where actual bytes could leak — the content parses as
nothing but CERTIFICATE PEM blocks that `ssl.SSLContext.load_verify_locations`
accepts. Any private key, CSR, or other PEM label, any path outside the
allowlist, or any content that fails to parse still classifies as
`credential`. The allowlist can only narrow what `.pem`/`.crt` reports as
forbidden, never widen it — see `_is_trusted_ca_path` and
`is_verified_ca_bundle`.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import ssl
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Callable

FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (".git/", "vcs_directory"),
    (".env", "env_file"),
    ("runtime/vectorstore/", "index_artifact"),
    ("runtime/documents/", "corpus_artifact"),
    ("runtime/index/versions/", "index_artifact"),
    ("models/intent_classifier/", "model_artifact"),
    (".ollama/", "ollama_data"),
    ("evaluation/reports/", "ci_report"),
    ("id_rsa", "credential"),
    (".pem", "credential"),
    (".pfx", "credential"),
    ("simple_qna_rag_test_seam", "test_embedding_seam"),
)

# Known OS/interpreter CA trust-store locations. A `.pem`/`.crt` member
# under one of these prefixes (or matching a suffix below) is a candidate
# for the allowlist carve-out — but path alone never exempts it; see
# is_verified_ca_bundle() for the content-side of the fail-closed check.
_TRUSTED_CA_PATH_PREFIXES: tuple[str, ...] = (
    "etc/ssl/certs/",              # Debian/Ubuntu system trust store
    "usr/lib/ssl/",                # OpenSSL default trust store location
    "usr/share/ca-certificates/",  # Debian/Ubuntu trust store source certs
    "etc/pki/tls/certs/",          # RHEL/CentOS/Fedora trust store
    "etc/pki/ca-trust/",           # RHEL/CentOS/Fedora trust anchors
)
_TRUSTED_CA_PATH_SUFFIXES: tuple[str, ...] = (
    "/certifi/cacert.pem",  # Python certifi package (top-level or vendored,
                             # e.g. pip/_vendor/certifi/cacert.pem)
)
_PEM_LABEL_RE = re.compile(rb"-----BEGIN ([A-Z0-9 ]+)-----")


def export_image(image: str, out_tar: Path) -> None:
    subprocess.run(["docker", "save", image, "-o", str(out_tar)], check=True)


def normalize_member_path(name: str) -> str:
    return posixpath.normpath(name.lstrip("/"))


def _is_trusted_ca_path(norm: str) -> bool:
    if not (norm.endswith(".pem") or norm.endswith(".crt")):
        return False
    if any(norm.startswith(prefix) for prefix in _TRUSTED_CA_PATH_PREFIXES):
        return True
    return any(norm.endswith(suffix) for suffix in _TRUSTED_CA_PATH_SUFFIXES)


def is_verified_ca_bundle(data: bytes) -> bool:
    """Fail-closed content check: True only if `data` is exclusively
    CERTIFICATE PEM blocks that ssl.SSLContext accepts as CA material.

    Any non-CERTIFICATE PEM label (private key, CSR, public key, ...), any
    block ssl rejects, or any decode failure returns False. A file with no
    PEM blocks at all also returns False rather than vacuously passing.
    """
    labels = _PEM_LABEL_RE.findall(data)
    if not labels:
        return False
    if any(label.strip() != b"CERTIFICATE" for label in labels):
        return False
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return False
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(cadata=text)
    except ssl.SSLError:
        return False
    return True


def classify_member(
    name: str,
    read_content: Callable[[], bytes] | None = None,
    *,
    is_symlink: bool = False,
) -> tuple[str, str] | None:
    """Classify a tar member path as forbidden, or None if clean.

    `read_content` is an optional zero-arg callable returning the member's
    file bytes, supplied by callers only for members that are already path
    candidates for the CA allowlist (see _is_trusted_ca_path) — it is never
    required for the traversal/generic-pattern checks below. `is_symlink`
    marks members that carry no content of their own (a symlink under a
    trusted CA path, e.g. usr/lib/ssl/cert.pem -> /etc/ssl/certs/... on
    Debian, leaks nothing by existing, so it is exempt without a content
    read; its symlink target, if itself a real file, is scanned as its own
    tar member).
    """
    norm = normalize_member_path(name)
    if norm.split("/", 1)[0] == "..":
        return ("path_traversal", name)
    if _is_trusted_ca_path(norm):
        if is_symlink:
            return None
        if read_content is not None:
            try:
                data = read_content()
            except (OSError, tarfile.TarError):
                data = None
            if data is not None and is_verified_ca_bundle(data):
                return None
    for pattern, category in FORBIDDEN_PATTERNS:
        if pattern.rstrip("/") in norm:
            return (category, pattern)
    return None


def is_whiteout(name: str) -> bool:
    base = posixpath.basename(normalize_member_path(name))
    return base.startswith(".wh.")


def scan(image: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "image.tar"
        export_image(image, archive)
        with tarfile.open(archive) as outer:
            manifest = json.loads(outer.extractfile("manifest.json").read())
            layer_paths = [entry for cfg in manifest for entry in cfg["Layers"]]
            violations = []
            layer_reports = []
            for layer_path in layer_paths:
                members = []
                with tarfile.open(fileobj=outer.extractfile(layer_path)) as layer:
                    for member in layer.getmembers():
                        if is_whiteout(member.name):
                            continue
                        read_content = None
                        if member.isfile():
                            def read_content(_layer=layer, _member=member) -> bytes:
                                fileobj = _layer.extractfile(_member)
                                return fileobj.read() if fileobj is not None else b""
                        hit = classify_member(
                            member.name, read_content, is_symlink=member.issym() or member.islnk()
                        )
                        if hit:
                            violations.append({
                                "layer": layer_path, "member": member.name,
                                "category": hit[0], "pattern": hit[1],
                            })
                        members.append(member.name)
                layer_reports.append({"layer": layer_path, "member_count": len(members)})
            return {
                "schema": "m43-layer-scan-v1", "image": image,
                "layers": layer_reports, "violations": violations,
                "forbidden_count": len(violations),
            }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    result = scan(args.image)
    text = json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["forbidden_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
