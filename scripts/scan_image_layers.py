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
narrowed the generic `.pem`/`.crt` credential patterns below with a
fail-closed CA allowlist: legitimate OS/interpreter trust-store bundles
(Debian `/etc/ssl/certs`, `/usr/lib/ssl`, `/usr/share/ca-certificates`,
RHEL `/etc/pki/...`, and any vendored `certifi/cacert.pem`) are exempted
only when the member is a genuine regular file AND its path matches a
known trust-store location AND its content is, byte for byte, nothing but
one or more complete CERTIFICATE PEM blocks that
`ssl.SSLContext.load_verify_locations` accepts as structurally valid.

Hosted-CI remediation iteration 2 (Code_Review_Iteration_3.md
CR-I3-MAJ-01/02) closed two allowlist gaps: `is_verified_ca_bundle` now
requires the *entire* byte stream to fullmatch a strict BEGIN/END
CERTIFICATE block grammar (rejecting appended/prepended secrets, mixed
PEM labels, and unmatched delimiters instead of merely scanning for a
`BEGIN` line), and `classify_member` grants the content exemption only to
`TarInfo.isfile()` members — a symlink, hardlink, device, or FIFO at a
trust-store-shaped path can no longer borrow the allowlist by path alone
and instead falls through to the generic forbidden-pattern check, which
now also covers `.crt`. Any private key, CSR, or other PEM label, any
path outside the allowlist, any non-regular member, or any content that
fails to parse still classifies as `credential`. The allowlist can only
narrow what `.pem`/`.crt` reports as forbidden, never widen it — see
`_is_trusted_ca_path` and `is_verified_ca_bundle`.
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
    (".crt", "credential"),
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
_WS = r"[ \t\r\n]*"
_B64_LINE = r"[A-Za-z0-9+/=]+"
_CERT_BLOCK = (
    r"-----BEGIN CERTIFICATE-----\r?\n"
    rf"(?:{_B64_LINE}\r?\n)+"
    r"-----END CERTIFICATE-----"
)
# Full-input fullmatch: the entire byte stream must be one or more complete
# BEGIN/END CERTIFICATE blocks separated by nothing but whitespace — any
# prepended/appended/interleaved text (secrets, other PEM labels, malformed
# bytes, unmatched delimiters) breaks the match and the file is rejected.
_STRICT_PEM_BUNDLE_RE = re.compile(rf"^{_WS}(?:{_CERT_BLOCK}{_WS})+$")


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
    """Fail-closed content check: True only if `data`, in its entirety, is
    one or more complete `-----BEGIN CERTIFICATE-----`/`-----END
    CERTIFICATE-----` blocks separated by nothing but whitespace, and the
    result parses as structurally valid X.509 material.

    `_STRICT_PEM_BUNDLE_RE.fullmatch` anchors both ends of the input, so
    every byte must belong to a block or to the permitted whitespace
    between blocks — arbitrary text prepended, appended, or interleaved
    (secrets, key=value payloads, a private-key/CSR/other PEM block, an
    unmatched or mismatched BEGIN/END delimiter, non-base64 bytes) leaves
    at least one byte unconsumed and fails the match. A file with no PEM
    blocks at all also returns False rather than vacuously passing.
    """
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return False
    if not _STRICT_PEM_BUNDLE_RE.fullmatch(text):
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
    is_regular_file: bool = True,
) -> tuple[str, str] | None:
    """Classify a tar member path as forbidden, or None if clean.

    `read_content` is an optional zero-arg callable returning the member's
    file bytes, supplied by callers only for members that are already path
    candidates for the CA allowlist (see _is_trusted_ca_path) — it is never
    required for the traversal/generic-pattern checks below.

    `is_regular_file` must reflect the tar member's actual type
    (`TarInfo.isfile()`/`isreg()`) and gates the CA-bundle content
    exemption: it is granted only to a genuine regular file, whose bytes
    can be read directly and structurally verified as a pure certificate
    bundle by `is_verified_ca_bundle`. A symlink, hardlink, device, FIFO,
    or directory at an otherwise-trusted CA path is never exempted by path
    alone — a hardlink shares another member's raw bytes with no
    verification of its own, and a symlink's target is a separate,
    independently-scanned tar member whose own classification is what
    actually gates it. Such non-regular members fall through to the
    generic forbidden-pattern check below, so a `.pem`/`.crt`-named link
    still classifies as `credential` even at a trust-store-shaped path.
    """
    norm = normalize_member_path(name)
    if norm.split("/", 1)[0] == "..":
        return ("path_traversal", name)
    if is_regular_file and _is_trusted_ca_path(norm) and read_content is not None:
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
                            member.name, read_content, is_regular_file=member.isfile()
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
