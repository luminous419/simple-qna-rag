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

Hosted-CI remediation iteration 3 (Code_Review_Iteration_4.md hosted run
31609022196) found the iteration-2 conservative link rejection produced
false positives against the *real* Debian base image: `/etc/ssl/certs/*`
is almost entirely `openssl rehash`-style symlinks (a numeric-hash link to
a CN-named link to a regular `.crt` under
`/usr/share/ca-certificates/...`), and `/usr/lib/ssl/cert.pem` is a
symlink to `/etc/ssl/certs/ca-certificates.crt`. It also found
`is_verified_ca_bundle`'s strict grammar rejected the *real* certifi
`cacert.pem`, whose upstream format interleaves each `BEGIN CERTIFICATE`
block with seven fixed `# Issuer:`/`# Subject:`/`# Label:`/`# Serial:`/
`# MD5 Fingerprint:`/`# SHA1 Fingerprint:`/`# SHA256 Fingerprint:`
comment lines. Both were confirmed against real exported image bytes
(`docker run` inspection of the built `production` target), not assumed.

The fix keeps both allowlist gates fail-closed while covering these real
shapes: `is_verified_ca_bundle`'s grammar accepts a certifi comment
stanza immediately before a `BEGIN CERTIFICATE` block — any other
comment text, or a comment anywhere else in the stream, still breaks the
full-consumption match, so a secret disguised as `# note: ...` is still
rejected. Separately, `classify_member` gained a `link_target_verified`
parameter: `scan()` now builds an OCI-union (whiteout-aware,
layer-ordered) merged filesystem state as it walks layers, and for a
symlink/hardlink at a trust-store path, `_resolve_trusted_link_content`
chases the link chain — bounded to `_MAX_LINK_HOPS` hops, cycle-detected
via a visited-path set, rejecting any hop that normalizes outside the
image root — and only reports a verified target when the chain lands on
a *genuine regular member* that is itself at a trust-store path (never
widening the allowlist to non-trust-store targets) whose bytes
independently pass `is_verified_ca_bundle`. A dangling target, a cycle, a
traversal/absolute escape, a non-regular final member, a whiteout-masked
path, or content that fails verification all still classify as
`credential` — the link itself is never trusted by path or name alone.

Code Review Iteration 5 (CR-I5-MAJ-01) found the iteration-3 comment
grammar above accepted *any number and order* of the seven prefixes with
*unbounded free-text values* (`(?:_COMMENT_LINE)*` with a
`[^\r\n]{0,512}` value class) — adversarial probes with duplicated,
reordered, or token/path/key-value-bearing "recognized-prefix" comments
all still matched. The grammar now encodes the exact upstream certifi
stanza as a single fixed, non-repeating sequence — `# Issuer:`, then
`# Subject:`, then `# Label:`, then `# Serial:`, then
`# MD5 Fingerprint:`, then `# SHA1 Fingerprint:`, then
`# SHA256 Fingerprint:`, each appearing at most once because the
sequence names each field literally exactly one time — so a missing,
duplicated, reordered, or extra field breaks the match instead of being
silently accepted or ignored. Each field also carries its own bounded
grammar derived from the real installed certifi package (every value
character actually observed across its full bundle, plus a generous
length margin): `# Serial:` is decimal digits only; the three
`Fingerprint:` fields are colon-separated lowercase hex of the exact
byte length for their algorithm (16/20/32 bytes); `# Issuer:`/
`# Subject:` accept only the RFC 4514-shaped
letters/digits/space/underscore/`()/,.=\\-` alphabet, and `# Label:` the
same minus underscore/comma/`/`/`=` (additionally requiring a wrapping
quoted string) that a Distinguished Name can legitimately contain —
colons, `@`, `$`, control characters, and other key=value/token/secret-
shaped punctuation are outside every field's alphabet and still break
the match. The stanza is optional as a whole (a system trust-store bundle
with no comments at all still verifies) but, when present, must be the
complete unbroken seven-field sequence described above.

Code Review Iteration 6 (CR-I6-MAJ-01) found the iteration-5 grammar,
while structurally exact, still let Issuer/Subject values be *any*
grammar-valid text — `CN=API_TOKEN=supersecret`, `CN=../../etc/shadow`,
`CN=PRIVATE KEY`, and `CN=AWS_SECRET_ACCESS_KEY=ABCDEF` all satisfy
`_ISSUER_SUBJECT_VALUE` without being derived from the certificate that
follows them, because the grammar only bounded the *shape* of the
comment text, never bound it to the certificate's own content. The fix
adds a second, independent gate layered on top of the grammar: every
accepted stanza is now parsed field-by-field and bound to the exact
certificate immediately following it, using only the stdlib —
`ssl.PEM_cert_to_DER_cert` for the DER bytes, `hashlib.md5`/`sha1`/
`sha256` over those bytes for the three fingerprints, and
`ssl._ssl._test_decode_cert` (the same internal decoder `ssl.py` itself
uses to test certificate parsing — invoked here against a private,
securely-created `tempfile.mkstemp` PEM file, since it only accepts a
filesystem path) for the Issuer/Subject RDN sequence and the serial
number. Issuer/Subject are re-rendered with `_canonical_dn` — CN, then
O, then every OU joined by `/`, each present only if that attribute type
actually exists in the certificate, with any non-ASCII character
`str.encode("unicode_escape")`-rendered exactly as certifi's own ASCII
comment output does — and the stanza value must equal that rendering
exactly. This canonicalization was proven, and is regression-tested,
against every one of the 147 entries in the locally installed certifi
bundle (`certifi==2025.10.5`), an unmodified copy of the real upstream
package, not a hand-built approximation. Serial is bound by comparing
the stanza's decimal digits to `str(int(decoded_hex_serial, 16))`.
Label is Mozilla-curated free text, not mechanically derived from the
DN by any single consistent rule (it is sometimes the Subject's CN,
sometimes an OU). For an ordinary entry, Label is bound by exact
byte-for-byte equality — no case-folding, no whitespace stripping — to
some individual RDN value (CN, O, or any single OU — never the merged
Issuer/Subject line) actually present in the certificate's own decoded
Subject. A Label that doesn't exactly match any real Subject RDN
value — including every token/path/key-value/private-material probe
that stays inside `_LABEL_VALUE`'s alphabet, and including a case-only
or leading/trailing-whitespace variant of a genuine RDN value — fails
this bind (CR-I7-MAJ-01). A small number of genuine upstream certifi
entries render a Label that differs from every Subject RDN value only
by case or a leading space, or (the legacy Entrust.net entry) isn't
derived from the Subject at all; each such entry is encoded as its own
exact `(certificate SHA-256, exact label string)` pair in
`_CERTIFICATE_LABEL_COMPATIBILITY` rather than as a case/whitespace
equivalence class, keeping every accepted exception certificate-bound
and non-extensible. A mismatch on any of the seven fields, or a
certificate block that fails to decode, rejects the whole bundle; the
independently parseable grammar above is retained as a first, cheaper
gate, and this certificate-binding gate is strictly additional to it,
never a replacement.

Code Review Iteration 8 (CR-I8-MAJ-01) found the Iteration 7 remediation's
three-entry exception table was tuned against one specific interpreter's
pip-vendored `certifi` copy and rejected the reviewer's own environment's
genuine pip-vendored bundle, whose pip shipped a different `certifi`
release (`2023.07.22`) than the one the remediation had checked
(`2026.7.22`). That gap exists because `pip`'s vendored `certifi` copy is
source baked into the installed `pip` package itself
(`pip/_vendor/certifi/cacert.pem`) — it is a separate, independent copy
from the top-level `certifi` package `requirements.lock` pins, is never
touched by installing or upgrading the top-level package, and instead
tracks whatever `pip` release happens to be installed, so it legitimately
differs release to release and interpreter to interpreter. The Iteration
7 remediation's claim that "both real bundles" were `certifi==2026.7.22`
was therefore wrong for the pip-vendored one: that version number only
ever applied to the top-level package, and the pip-vendored copy's actual
version was never independently checked.

The supported compatibility boundary is now explicit and spans three real
bundles across the two interpreters this project's tests are actually run
under:

- This project's maintained `venv` — `python -m pytest` — where the
  top-level `certifi` package is pinned by `requirements.lock`
  (`certifi==2026.7.22` at the time of writing) and `pip`'s vendored copy
  (`pip._vendor.certifi`) tracks whichever `pip` release was installed
  into that `venv`.
- This machine's repository-default Python (`python3`/`pytest` resolved
  from `PATH` without activating `venv`) — the interpreter
  Code_Review_Iteration_8.md's fresh review actually ran under, whose
  `pip==23.3.1` vendors `certifi==2023.07.22`. Running the module's tests
  under this interpreter is a genuine, reproducible local-review
  configuration, not a one-off review artifact, so its pip-vendored
  bundle is a supported boundary in its own right rather than an
  out-of-scope historical curiosity.

`_CERTIFICATE_LABEL_COMPATIBILITY` covers every ordinary-Label deviation
an independent walk of all three bundles finds: the legacy Entrust.net
entry and `certSIGN Root CA G2` (both interpreters' bundles),
` OISTE Server Root RSA G1` (the `venv` pip-vendored bundle), and — from
the repository-default interpreter's `certifi==2023.07.22` pip-vendored
bundle — Comodo AAA Services root, Security Communication Root CA, XRamp
Global CA Root, Go Daddy Class 2 CA, and Starfield Class 2 CA. Mozilla has
since removed all five of those from the curated root list, so none of
them (by SHA-256) is present in either bundle the `venv` interpreter
exercises today; each is nevertheless a real, unmodified, historical
certifi entry, verified directly against `_label_bound_to_subject` with
its exact SHA-256/Label pair and a genuine decoded Subject RDN value from
that same certificate (see
tests/unit/test_scan_image_layers.py's CR-I8-MAJ-01 section) — the table's
correctness for them does not depend on running under that exact
interpreter every time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

# The upstream certifi cacert.pem (github.com/certifi/python-certifi,
# generated from Mozilla's included-cert list) precedes every certificate
# with exactly these seven metadata fields, in this exact order, one per
# line, each exactly once (confirmed against the real installed package:
# every one of its blocks has all seven, in this order, no duplicates).
# Each field's value alphabet/length below is derived from every value
# actually observed across the full real bundle, with a generous margin —
# not an arbitrary text allowance. Because each field name below appears
# literally exactly once in the fixed sequence (never inside a repeating
# group), a stanza that is missing a field, duplicates a field, reorders
# the fields, or appends an extra field cannot match: the parser expects
# the next literal field name and finds either the wrong one or the
# `-----BEGIN CERTIFICATE-----` marker instead.
_HEX_BYTE = r"[0-9a-f]{2}"
# Serial: decimal only. Real bundle values are 1-48 digits (no sign, no
# leading text) — RFC 5280 serials are at most 20 octets, so 48 digits is
# already a wide margin over any conformant certificate.
_SERIAL_VALUE = r"[0-9]{1,48}"
# Fingerprint: colon-separated lowercase hex, byte-exact for the named
# digest (MD5=16 bytes, SHA1=20 bytes, SHA256=32 bytes) — not a free-form
# hex blob of arbitrary length.
_MD5_FINGERPRINT_VALUE = rf"(?:{_HEX_BYTE}:){{15}}{_HEX_BYTE}"
_SHA1_FINGERPRINT_VALUE = rf"(?:{_HEX_BYTE}:){{19}}{_HEX_BYTE}"
_SHA256_FINGERPRINT_VALUE = rf"(?:{_HEX_BYTE}:){{31}}{_HEX_BYTE}"
# Issuer/Subject: the real bundle's DN-rendering alphabet is letters,
# digits, space, underscore, and `()/,.=\-` (no colon, `@`, `$`, quote,
# or control byte — none of which a rendered X.509 DN needs but all of
# which a smuggled token/key-value/secret typically does). Underscore is
# included because a real DN legitimately contains it (confirmed against
# two independently-vendored real certifi copies in the built production
# image — pip's vendored `certifi==2025.10.5`'s Entrust.net 2048 CA entry
# renders `OU=...CPS_2048...`; excluding it produced a false-positive
# reject of that genuine trust-store bundle, CR-I5-MAJ-01 remediation
# verification). Real observed max length is 148 bytes; 256 is a wide
# margin.
_ISSUER_SUBJECT_VALUE = r"[A-Za-z0-9 ()/,._=\\-]{1,256}"
# Label: a quoted string using the same narrow alphabet minus the comma
# (never observed inside a real Label) and minus `/` and `=` (never
# observed either — Labels are short human-readable names, not DNs).
# Real observed max length (incl. quotes) is 61 bytes; 128 is a wide
# margin.
_LABEL_VALUE = r'"[A-Za-z0-9 ().\\-]{0,126}"'
_CERTIFI_STANZA = (
    rf"# Issuer: {_ISSUER_SUBJECT_VALUE}\r?\n"
    rf"# Subject: {_ISSUER_SUBJECT_VALUE}\r?\n"
    rf"# Label: {_LABEL_VALUE}\r?\n"
    rf"# Serial: {_SERIAL_VALUE}\r?\n"
    rf"# MD5 Fingerprint: {_MD5_FINGERPRINT_VALUE}\r?\n"
    rf"# SHA1 Fingerprint: {_SHA1_FINGERPRINT_VALUE}\r?\n"
    rf"# SHA256 Fingerprint: {_SHA256_FINGERPRINT_VALUE}\r?\n"
)
_CERT_BLOCK = (
    rf"(?:{_CERTIFI_STANZA})?"
    r"-----BEGIN CERTIFICATE-----\r?\n"
    rf"(?:{_B64_LINE}\r?\n)+"
    r"-----END CERTIFICATE-----"
)
# Full-input fullmatch: the entire byte stream must be one or more complete
# BEGIN/END CERTIFICATE blocks (each optionally preceded by recognized
# certifi metadata comment lines) separated by nothing but whitespace — any
# prepended/appended/interleaved text (secrets, other PEM labels, malformed
# bytes, unmatched delimiters, unrecognized comments) breaks the match and
# the file is rejected.
_STRICT_PEM_BUNDLE_RE = re.compile(rf"^{_WS}(?:{_CERT_BLOCK}{_WS})+$")

# Named-group variant of the same stanza+cert grammar, used only to walk
# the (already fullmatch-confirmed) bundle block by block so each accepted
# stanza can be bound to its own immediately-following certificate — see
# CR-I6-MAJ-01 in the module docstring. The value grammars are identical to
# `_CERT_BLOCK` above; this is purely an iteration aid, not a separate or
# looser acceptance grammar.
_CERTIFI_STANZA_GROUPED = (
    rf"# Issuer: (?P<issuer>{_ISSUER_SUBJECT_VALUE})\r?\n"
    rf"# Subject: (?P<subject>{_ISSUER_SUBJECT_VALUE})\r?\n"
    rf"# Label: (?P<label>{_LABEL_VALUE})\r?\n"
    rf"# Serial: (?P<serial>{_SERIAL_VALUE})\r?\n"
    rf"# MD5 Fingerprint: (?P<md5>{_MD5_FINGERPRINT_VALUE})\r?\n"
    rf"# SHA1 Fingerprint: (?P<sha1>{_SHA1_FINGERPRINT_VALUE})\r?\n"
    rf"# SHA256 Fingerprint: (?P<sha256>{_SHA256_FINGERPRINT_VALUE})\r?\n"
)
_CERT_BLOCK_GROUPED_RE = re.compile(
    rf"(?:{_CERTIFI_STANZA_GROUPED})?"
    r"(?P<pem>-----BEGIN CERTIFICATE-----\r?\n"
    rf"(?:{_B64_LINE}\r?\n)+"
    r"-----END CERTIFICATE-----)"
)

# Attribute-type OIDs `ssl._ssl._test_decode_cert` names in an RDN, in the
# exact preference certifi's own stanza rendering uses: Common Name first,
# then Organization, then every Organizational Unit (joined by `/` when
# there is more than one) — never Country, Locality, State, or any other
# attribute type, none of which the real installed bundle's stanzas ever
# render (confirmed against all 147 entries).
_DN_COMMON_NAME = "commonName"
_DN_ORGANIZATION = "organizationName"
_DN_ORG_UNIT = "organizationalUnitName"


def _unicode_escape(value: str) -> str:
    """Render `value` the way certifi's own ASCII-only comment stanzas
    render a non-ASCII DN character: Python's `unicode_escape` codec
    (`\\xHH` for a byte in 0x80-0xFF, `\\uHHHH` above that), leaving every
    printable-ASCII character — including the `=`, `/`, and space that
    join the rendered DN — untouched. Confirmed byte-for-byte against the
    one non-ASCII entry (NetLock, containing Hungarian accented letters)
    in the locally installed certifi bundle."""
    return value.encode("unicode_escape").decode("ascii")


def _canonical_dn(rdns: tuple) -> str:
    """Re-render a decoded X.509 RDN sequence (the `issuer`/`subject`
    shape `ssl._ssl._test_decode_cert` returns: a tuple of RDNs, each a
    tuple of `(attribute_type, value)` pairs) in the exact format upstream
    certifi's stanza comments use for that same certificate: `CN=...`,
    then `O=...`, then `OU=...` with every organizational-unit value
    joined by `/`, each part present only if the certificate actually
    carries that attribute type. Verified byte-for-byte against all 147
    Issuer and 147 Subject values in the locally installed certifi
    bundle — see `test_is_verified_ca_bundle_binds_full_installed_bundle_stanzas_to_certificates`.
    """
    flat = [(attr_type, value) for rdn in rdns for attr_type, value in rdn]
    common_name = next((v for t, v in flat if t == _DN_COMMON_NAME), None)
    organization = next((v for t, v in flat if t == _DN_ORGANIZATION), None)
    org_units = [v for t, v in flat if t == _DN_ORG_UNIT]
    parts = []
    if common_name is not None:
        parts.append(f"CN={common_name}")
    if organization is not None:
        parts.append(f"O={organization}")
    if org_units:
        parts.append(f"OU={'/'.join(org_units)}")
    return _unicode_escape(" ".join(parts))


def _canonical_serial(hex_serial: str) -> str:
    """`ssl._ssl._test_decode_cert`'s `serialNumber` is unsigned hex with
    no separators (e.g. `'456B5054'`); certifi's stanza renders the same
    integer in decimal — this is the inverse of that formatting choice,
    not an independent guess at the value."""
    return str(int(hex_serial, 16))


def _colonize_hex(hex_digest: str) -> str:
    """`hashlib.*.hexdigest()` returns a contiguous lowercase hex string;
    certifi's fingerprint stanza fields are the same bytes colon-separated
    two hex digits at a time."""
    return ":".join(hex_digest[i : i + 2] for i in range(0, len(hex_digest), 2))


_CERTIFICATE_LABEL_COMPATIBILITY: dict[str, str] = {
    # Genuine legacy Mozilla label whose wording is not present in any
    # Subject RDN, not even case-insensitively.
    "6dc47172e01cbcb0bf62580d895fe2b8ac9ad4f873801e0c10b9c837d21eb177":
        "Entrust.net Premium 2048 Secure Server CA",
    # Real entry: Label renders lowercase-leading `certSIGN Root CA G2`
    # against a Subject CN of literal `certSIGN ROOT CA G2` — case differs,
    # confirmed against the locally installed and pip-vendored certifi
    # bundles (certifi==2026.7.22).
    "657cfe2fa73faa38462571f332a2363a46fce7020951710702cdfbb6eeda3305":
        "certSIGN Root CA G2",
    # Real entry: Label carries a genuine leading space, ` OISTE Server
    # Root RSA G1`, against a Subject CN with none — confirmed against the
    # locally installed and pip-vendored certifi bundles.
    "9ae36232a5189ffddb353dfd26520c015395d22777dac59db57b98c089a651e6":
        " OISTE Server Root RSA G1",
    # CR-I8-MAJ-01: five further genuine legacy entries, each verified
    # against the real, unmodified certifi==2023.07.22 pip-vendored bundle
    # this machine's repository-default Python resolves (see the module
    # docstring's three-boundary write-up) via direct
    # _label_bound_to_subject calls — see the
    # test_label_bound_to_subject_accepts_certifi_2023_legacy_exception /
    # _rejects_mutated_legacy_label / _rejects_legacy_label_with_wrong_sha256
    # tests in tests/unit/test_scan_image_layers.py. None of these five
    # certificates (by SHA-256) is present in either bundle the venv's
    # full-bundle tests exercise; Mozilla has since removed all five from
    # the curated root list.
    "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4":
        "Comodo AAA Services root",
    "e75e72ed9f560eec6eb4800073a43fc3ad19195a392282017895974a99026b6c":
        "Security Communication Root CA",
    "cecddc905099d8dadfc5b1d209b737cbe2c18cfb2c10c0ff0bcf0d3286fc1aa2":
        "XRamp Global CA Root",
    "c3846bf24b9e93ca64274c0ec67c1ecc5e024ffcacd2d74019350e81fe546ae4":
        "Go Daddy Class 2 CA",
    "1465fa205397b876faa6f0a9958e5590e40fcc7faa4fb7c2c8677521fb5fb658":
        "Starfield Class 2 CA",
}


def _label_bound_to_subject(
    label_value: str, subject_rdns: tuple, certificate_sha256: str
) -> bool:
    """certifi's `# Label:` is Mozilla-curated descriptive text, not a
    value mechanically derived from the DN by one consistent rule (it is
    the Subject's CN for most entries, an Organizational Unit for others).
    For an ordinary entry the accepted Label must equal — byte-for-byte,
    no case-folding, no whitespace stripping — an individual RDN value
    (CN, O, or one OU) the certificate's own decoded Subject actually
    carries. It is never matched against the merged Issuer/Subject stanza
    line, so a Label that isn't an exact real Subject RDN value —
    including every grammar-valid token/path/key-value/private-material
    probe, and including a case-only or leading/trailing-whitespace
    variant of a genuine RDN value — fails this bind.

    A small number of genuine upstream certifi entries render a Label
    that differs from every Subject RDN value by case or a leading space
    (`certSIGN Root CA G2` vs. Subject CN `certSIGN ROOT CA G2`;
    ` OISTE Server Root RSA G1`, with a real leading space, vs. a Subject
    CN with none) or isn't derived from the Subject at all (the legacy
    Entrust.net label). Requiring exact RDN equality would
    false-positive-reject those genuine, unmodified entries, so each is
    instead encoded as its own exact `(certificate SHA-256, exact label
    string)` pair in `_CERTIFICATE_LABEL_COMPATIBILITY` — bound to one
    specific certificate and one specific byte-exact label, never a
    case/whitespace-insensitive class of labels. CR-I7-MAJ-01."""
    candidates = {_unicode_escape(value) for rdn in subject_rdns for _, value in rdn}
    if label_value in candidates:
        return True
    compatible_label = _CERTIFICATE_LABEL_COMPATIBILITY.get(certificate_sha256)
    return compatible_label is not None and label_value == compatible_label


def _decode_certificate(pem_text: str) -> dict:
    """Decode a single `-----BEGIN CERTIFICATE-----` PEM block using only
    the stdlib. `ssl._ssl._test_decode_cert` — the same internal decoder
    `ssl.py` itself uses to test certificate parsing — only accepts a
    filesystem path, so the block is written to a private, securely
    created (`tempfile.mkstemp`, mode 0600, unique name) temporary file for
    the duration of the call and removed immediately after, regardless of
    outcome. No third-party ASN.1/X.509 library is used."""
    fd, tmp_path = tempfile.mkstemp(suffix=".pem")
    try:
        handle = os.fdopen(fd, "w", encoding="ascii")
        fd = -1  # ownership transferred to `handle`
        with handle:
            handle.write(pem_text)
        return ssl._ssl._test_decode_cert(tmp_path)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _stanza_bound_to_certificate(stanza: dict, pem_text: str) -> bool:
    """CR-I6-MAJ-01: an accepted seven-field stanza must equal values
    independently derived from the exact certificate immediately following
    it — an otherwise grammar-valid Issuer/Subject/Label comment that
    isn't actually the metadata for that certificate (e.g.
    `# Issuer: CN=API_TOKEN=supersecret` preceding a real, unrelated CA
    certificate) fails this check even though it satisfies every field's
    character-class grammar. Returns False on any single-field mismatch or
    if the certificate itself fails to decode; never partially trusts a
    stanza."""
    try:
        decoded = _decode_certificate(pem_text)
    except ssl.SSLError:
        return False
    der = ssl.PEM_cert_to_DER_cert(pem_text)
    sha256 = hashlib.sha256(der).hexdigest()
    return (
        stanza["issuer"] == _canonical_dn(decoded["issuer"])
        and stanza["subject"] == _canonical_dn(decoded["subject"])
        and stanza["serial"] == _canonical_serial(decoded["serialNumber"])
        and stanza["md5"] == _colonize_hex(hashlib.md5(der).hexdigest())
        and stanza["sha1"] == _colonize_hex(hashlib.sha1(der).hexdigest())
        and stanza["sha256"] == _colonize_hex(sha256)
        and _label_bound_to_subject(
            stanza["label"].strip('"'), decoded["subject"], sha256
        )
    )


def _all_stanzas_bound_to_certificates(text: str) -> bool:
    """Walk every block of an already fullmatch-confirmed bundle and, for
    each one that carries a stanza, bind it to its own certificate. A
    bundle with no stanzas at all (a bare system trust-store file) has
    nothing to bind and passes trivially — this gate only ever narrows
    what the grammar above accepts, never widens it."""
    for match in _CERT_BLOCK_GROUPED_RE.finditer(text):
        if match.group("issuer") is None:
            continue
        if not _stanza_bound_to_certificate(match.groupdict(), match.group("pem")):
            return False
    return True


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
    return _all_stanzas_bound_to_certificates(text)


def classify_member(
    name: str,
    read_content: Callable[[], bytes] | None = None,
    *,
    is_regular_file: bool = True,
    is_link: bool = False,
    link_target_verified: bool = False,
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
    bundle by `is_verified_ca_bundle`.

    `is_link` (`TarInfo.issym()` or `TarInfo.islnk()`) and
    `link_target_verified` gate a second, independent exemption for a
    symlink/hardlink at a trust-store path: this function never resolves
    or reads the link's target itself (a hardlink shares another member's
    raw bytes and a symlink's target is a separate, independently-scanned
    tar member), so `link_target_verified=True` must come from the
    caller having already performed bounded, cycle-safe, non-traversing
    resolution against the actual OCI layer state (see `scan()` and
    `_resolve_trusted_link_content`) and independently verified the
    resolved regular member's bytes. Absent that pre-verified evidence
    (the default for both parameters), a symlink, hardlink, device, FIFO,
    or directory at an otherwise-trusted CA path is never exempted by path
    alone and falls through to the generic forbidden-pattern check below,
    so a `.pem`/`.crt`-named link still classifies as `credential` even at
    a trust-store-shaped path.
    """
    norm = normalize_member_path(name)
    if norm.split("/", 1)[0] == "..":
        return ("path_traversal", name)
    if _is_trusted_ca_path(norm):
        if is_regular_file and read_content is not None:
            try:
                data = read_content()
            except (OSError, tarfile.TarError):
                data = None
            if data is not None and is_verified_ca_bundle(data):
                return None
        elif is_link and link_target_verified:
            return None
    for pattern, category in FORBIDDEN_PATTERNS:
        if pattern.rstrip("/") in norm:
            return (category, pattern)
    return None


_MAX_LINK_HOPS = 40  # generous bound, matching typical OS ELOOP limits


class _MergedEntry:
    """One path's state in the OCI-union (whiteout-aware) merged view of
    the image filesystem as of a given point in the layer stack."""

    __slots__ = ("kind", "linkname", "layer_path", "member_name")

    def __init__(self, kind: str, linkname: str, layer_path: str, member_name: str) -> None:
        self.kind = kind  # "reg" | "symlink" | "hardlink" | "other"
        self.linkname = linkname
        self.layer_path = layer_path
        self.member_name = member_name


def _update_merged_state(
    merged_state: dict[str, _MergedEntry], layer_path: str, members: list[tarfile.TarInfo]
) -> None:
    """Apply one layer's members to `merged_state` in place, OCI-union
    semantics: this layer's own whiteouts are resolved against the state
    inherited from earlier layers first, then this layer's own
    regular/symlink/hardlink/other writes are recorded — so a write always
    wins over a same-layer whiteout of the same path, and a write is
    visible even inside a directory the same layer opaque-whites out.
    Whiteout markers themselves are never entries in the merged state.
    """
    opaque_dirs: list[str] = []
    exact_deletes: list[str] = []
    writes: list[tuple[str, tarfile.TarInfo]] = []
    for member in members:
        norm = normalize_member_path(member.name)
        base = posixpath.basename(norm)
        if base == ".wh..wh..opq":
            opaque_dirs.append(posixpath.dirname(norm))
        elif base.startswith(".wh."):
            exact_deletes.append(
                normalize_member_path(posixpath.join(posixpath.dirname(norm), base[len(".wh."):]))
            )
        else:
            writes.append((norm, member))

    for directory in opaque_dirs:
        prefix = f"{directory}/" if directory else ""
        for path in list(merged_state):
            if directory == "" or path.startswith(prefix):
                del merged_state[path]
    for path in exact_deletes:
        merged_state.pop(path, None)

    for norm, member in writes:
        if member.isfile():
            kind = "reg"
        elif member.issym():
            kind = "symlink"
        elif member.islnk():
            kind = "hardlink"
        else:
            kind = "other"
        merged_state[norm] = _MergedEntry(kind, member.linkname, layer_path, member.name)


def _resolve_trusted_link_content(
    merged_state: dict[str, _MergedEntry],
    read_layer_member: Callable[[str, str], bytes | None],
    start_norm: str,
    linkname: str,
    is_hardlink: bool,
) -> bytes | None:
    """Bounded, cycle-safe resolution of a symlink/hardlink chain rooted at
    an already-trusted-path member. Returns the ultimate regular member's
    raw bytes only if every hop stays inside the image root, never repeats
    a path (no cycles), resolves within `_MAX_LINK_HOPS`, and lands on a
    genuine regular member recorded in the merged OCI filesystem state
    whose own normalized path is *also* a trust-store path — this never
    widens the allowlist to a target outside the recognized locations, it
    only lets a trust-store-internal link (e.g. Debian's
    `etc/ssl/certs/<hash>.0 -> <name>.pem -> /usr/share/ca-certificates/...`
    chain) reach the regular member whose bytes actually get verified.
    Returns None on any dangling target, cycle, traversal/absolute escape,
    non-regular or untrusted-path final member, or whiteout-masked path —
    the caller always falls back to the fail-closed generic pattern check.
    """
    current_norm = start_norm
    current_link = linkname
    current_is_hardlink = is_hardlink
    visited = {start_norm}
    for _ in range(_MAX_LINK_HOPS):
        if current_is_hardlink:
            # Tar hardlinks name another member of the same archive by its
            # own archive path, not a symlink-style path relative to the
            # link's parent directory.
            target_norm = normalize_member_path(current_link)
        elif current_link.startswith("/"):
            target_norm = normalize_member_path(current_link)
        else:
            target_norm = normalize_member_path(
                posixpath.join(posixpath.dirname(current_norm), current_link)
            )
        if target_norm.split("/", 1)[0] == "..":
            return None  # absolute-root or relative traversal escape
        if target_norm in visited:
            return None  # cycle
        visited.add(target_norm)

        entry = merged_state.get(target_norm)
        if entry is None:
            return None  # dangling, including whiteout-masked paths

        if entry.kind == "reg":
            if not _is_trusted_ca_path(target_norm):
                return None
            return read_layer_member(entry.layer_path, entry.member_name)
        if entry.kind == "symlink":
            current_norm, current_link, current_is_hardlink = target_norm, entry.linkname, False
            continue
        if entry.kind == "hardlink":
            current_norm, current_link, current_is_hardlink = target_norm, entry.linkname, True
            continue
        return None  # device/FIFO/directory target — never verifiable
    return None  # hop budget exhausted


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
            # Cumulative OCI-union filesystem state, layers [0..i] applied
            # in order — built incrementally so a trust-store symlink can
            # be resolved against every layer up to and including its own
            # (never a later one), matching real union-filesystem semantics
            # and never using information that would not yet exist at that
            # point in the image build.
            merged_state: dict[str, _MergedEntry] = {}
            for layer_path in layer_paths:
                with tarfile.open(fileobj=outer.extractfile(layer_path)) as layer:
                    members = layer.getmembers()
                    _update_merged_state(merged_state, layer_path, members)
                    member_index = {m.name: m for m in members}

                    def read_layer_member(
                        target_layer_path: str,
                        target_member_name: str,
                        _layer=layer,
                        _layer_path=layer_path,
                        _index=member_index,
                        _outer=outer,
                    ) -> bytes | None:
                        if target_layer_path == _layer_path:
                            info = _index.get(target_member_name)
                            if info is None:
                                return None
                            fileobj = _layer.extractfile(info)
                            return fileobj.read() if fileobj is not None else None
                        other_fileobj = _outer.extractfile(target_layer_path)
                        if other_fileobj is None:
                            return None
                        with tarfile.open(fileobj=other_fileobj) as other_layer:
                            try:
                                info = other_layer.getmember(target_member_name)
                            except KeyError:
                                return None
                            target_fileobj = other_layer.extractfile(info)
                            return target_fileobj.read() if target_fileobj is not None else None

                    for member in members:
                        if is_whiteout(member.name):
                            continue
                        read_content = None
                        if member.isfile():
                            def read_content(_layer=layer, _member=member) -> bytes:
                                fileobj = _layer.extractfile(_member)
                                return fileobj.read() if fileobj is not None else b""
                        is_link = member.issym() or member.islnk()
                        link_target_verified = False
                        if is_link and _is_trusted_ca_path(normalize_member_path(member.name)):
                            target_bytes = _resolve_trusted_link_content(
                                merged_state,
                                read_layer_member,
                                normalize_member_path(member.name),
                                member.linkname,
                                member.islnk(),
                            )
                            link_target_verified = (
                                target_bytes is not None and is_verified_ca_bundle(target_bytes)
                            )
                        hit = classify_member(
                            member.name,
                            read_content,
                            is_regular_file=member.isfile(),
                            is_link=is_link,
                            link_target_verified=link_target_verified,
                        )
                        if hit:
                            violations.append({
                                "layer": layer_path, "member": member.name,
                                "category": hit[0], "pattern": hit[1],
                            })
                    member_count = sum(1 for m in members if not is_whiteout(m.name))
                layer_reports.append({"layer": layer_path, "member_count": member_count})
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
