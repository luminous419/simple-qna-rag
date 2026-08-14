"""M4.3-REQ-005.4 — layer scanner positive/negative/traversal/whiteout fixtures."""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from scripts import scan_image_layers as scanner


def _make_layer_tar(names: list[str]) -> tarfile.TarFile:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in names:
            data = b"x"
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return tarfile.open(fileobj=buf)


def test_forbidden_layer_detects_multiple_categories():
    layer = _make_layer_tar([".git/HEAD", "runtime/vectorstore/index.faiss", "id_rsa"])
    violations = []
    for member in layer.getmembers():
        hit = scanner.classify_member(member.name)
        if hit:
            violations.append(hit[0])
    assert set(violations) == {"vcs_directory", "index_artifact", "credential"}


def test_clean_layer_has_no_violations():
    layer = _make_layer_tar(["src/simple_qna_rag/__init__.py", "README.md"])
    for member in layer.getmembers():
        assert scanner.classify_member(member.name) is None


def test_clean_web_asset_layer_has_no_violations():
    layer = _make_layer_tar(["web/static/style.css", "web/templates/index.html"])
    for member in layer.getmembers():
        assert scanner.classify_member(member.name) is None


def test_traversal_member_detected():
    layer = _make_layer_tar(["../../etc/passwd"])
    member = layer.getmembers()[0]
    hit = scanner.classify_member(member.name)
    assert hit is not None
    assert hit[0] == "path_traversal"


def test_whiteout_only_layer_has_zero_violations():
    layer = _make_layer_tar([".wh..wh..opq", "runtime/.wh.vectorstore"])
    violations = []
    for member in layer.getmembers():
        if scanner.is_whiteout(member.name):
            continue
        hit = scanner.classify_member(member.name)
        if hit:
            violations.append(hit)
    assert violations == []


def test_test_seam_leak_layer_detected():
    layer = _make_layer_tar(["tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py"])
    member = layer.getmembers()[0]
    hit = scanner.classify_member(member.name)
    assert hit is not None
    assert hit[0] == "test_embedding_seam"


def test_positive_negative_traversal_whiteout_fixtures():
    """Single node id referenced by run_m43_acceptance.py's PROFILE_NODE_IDS
    — exercises the five fixture categories above in one collected test."""
    test_forbidden_layer_detects_multiple_categories()
    test_clean_layer_has_no_violations()
    test_traversal_member_detected()
    test_whiteout_only_layer_has_zero_violations()
    test_test_seam_leak_layer_detected()


# --- Hosted-CI remediation iteration 1: CA trust-store allowlist ----------
#
# scripts/scan_image_layers.py's `.pem` pattern used to flag every OS and
# certifi CA bundle in the base image as a "credential" (153 false
# positives on PR #18 run 31593816593: Debian /etc/ssl/certs/*.pem,
# /usr/lib/ssl/cert.pem, and both the top-level and pip-vendored
# certifi/cacert.pem). classify_member() now only exempts a `.pem`/`.crt`
# member when BOTH its path sits under a known trust-store location AND
# its content verifies as nothing but CERTIFICATE PEM blocks accepted by
# ssl.SSLContext.load_verify_locations — everything below proves that gate
# stays fail-closed in both directions (real CA content still needs the
# right path, and the right path still needs real CA content).

_REAL_CA_CERT_PEM = b"""-----BEGIN CERTIFICATE-----
MIIEkTCCA3mgAwIBAgIERWtQVDANBgkqhkiG9w0BAQUFADCBsDELMAkGA1UEBhMC
VVMxFjAUBgNVBAoTDUVudHJ1c3QsIEluYy4xOTA3BgNVBAsTMHd3dy5lbnRydXN0
Lm5ldC9DUFMgaXMgaW5jb3Jwb3JhdGVkIGJ5IHJlZmVyZW5jZTEfMB0GA1UECxMW
KGMpIDIwMDYgRW50cnVzdCwgSW5jLjEtMCsGA1UEAxMkRW50cnVzdCBSb290IENl
cnRpZmljYXRpb24gQXV0aG9yaXR5MB4XDTA2MTEyNzIwMjM0MloXDTI2MTEyNzIw
NTM0MlowgbAxCzAJBgNVBAYTAlVTMRYwFAYDVQQKEw1FbnRydXN0LCBJbmMuMTkw
NwYDVQQLEzB3d3cuZW50cnVzdC5uZXQvQ1BTIGlzIGluY29ycG9yYXRlZCBieSBy
ZWZlcmVuY2UxHzAdBgNVBAsTFihjKSAyMDA2IEVudHJ1c3QsIEluYy4xLTArBgNV
BAMTJEVudHJ1c3QgUm9vdCBDZXJ0aWZpY2F0aW9uIEF1dGhvcml0eTCCASIwDQYJ
KoZIhvcNAQEBBQADggEPADCCAQoCggEBALaVtkNC+sZtKm9I35RMOVcF7sN5EUFo
Nu3s/poBj6E4KPz3EEZmLk0eGrEaTsbRwJWIsMn/MYszA9u3g3s+IIRe7bJWKKf4
4LlAcTfFy0cOlypowCKVYhXbR9n10Cv/gkvJrT7eTNuQgFA/CYqEAOwwCj0Yzfv9
KlmaI5UXLEWeH25DeW0MXJj+SKfFI0dcXv1u5x609mhF0YaDW6KKjbHjKYD+JXGI
rb68j6xSlkuqUY3kEzEZ6E5Nn9uss2rVvDlUccp6en+Q3X0dgNmBu1kmwhH+5pPi
94DkZfs0Nw4pgHBNrziGLp5/V6+eF67rHMsoIV+2HNjnogQi+dPa2MsCAwEAAaOB
sDCBrTAOBgNVHQ8BAf8EBAMCAQYwDwYDVR0TAQH/BAUwAwEB/zArBgNVHRAEJDAi
gA8yMDA2MTEyNzIwMjM0MlqBDzIwMjYxMTI3MjA1MzQyWjAfBgNVHSMEGDAWgBRo
kORnpKZTgMeGZqTx90tD+4S9bTAdBgNVHQ4EFgQUaJDkZ6SmU4DHhmak8fdLQ/uE
vW0wHQYJKoZIhvZ9B0EABBAwDhsIVjcuMTo0LjADAgSQMA0GCSqGSIb3DQEBBQUA
A4IBAQCT1DCw1wMgKtD5Y+iRDAUgqV8ZyntyTtSx29CW+1RaGSwMCPeyvIWonX9t
O1KzKtvn1ISMY/YPyyYBkVBs9F8U4pN0wBOeMDpQ47RgxRzwIkSNcUesyBrJ6Zua
AGAT/3B+XxFNSRuzFVJ7yVTav52Vr2ua2J7p8eRDjeIRRDq/r72DQnNSi6q7pynP
9WQcCk3RvKqsnyrQ/39/2n3qse0wJcGE2jTSW3iDVuycNsMm4hH2Z0kdkquM++v/
eu6FSqdQgPCnXEqULl8FmTxSQeDNtGPPAUO6nIPcj2A781q0tHuu2guQOHXvgR1m
0vdXcDazv/wor3ElhVsT/h5/WrQ8
-----END CERTIFICATE-----
"""

_FAKE_PRIVATE_KEY_PEM = b"""-----BEGIN RSA PRIVATE KEY-----
MIIBOgIBAAJBAK5FakePrivateKeyContentThatIsNotARealKeyButLooksLikeOneToASubstringMatcher
-----END RSA PRIVATE KEY-----
"""

_MALFORMED_CERT_PEM = b"""-----BEGIN CERTIFICATE-----
VGhpcyBpcyBub3QgYSByZWFsIGNlcnRpZmljYXRl
-----END CERTIFICATE-----
"""


def _read_member_content(tar: tarfile.TarFile, name: str) -> bytes:
    fileobj = tar.extractfile(name)
    assert fileobj is not None
    return fileobj.read()


def test_system_ca_pem_under_trust_store_path_is_allowed():
    layer = _make_layer_tar(["etc/ssl/certs/Entrust_Root_CA_G2.pem"])
    layer_with_content = io.BytesIO()
    with tarfile.open(fileobj=layer_with_content, mode="w") as tf:
        info = tarfile.TarInfo(name="etc/ssl/certs/Entrust_Root_CA_G2.pem")
        info.size = len(_REAL_CA_CERT_PEM)
        tf.addfile(info, io.BytesIO(_REAL_CA_CERT_PEM))
    layer_with_content.seek(0)
    with tarfile.open(fileobj=layer_with_content) as tf:
        member = tf.getmembers()[0]
        hit = scanner.classify_member(
            member.name, lambda: _read_member_content(tf, member.name)
        )
    assert hit is None
    del layer


def test_certifi_cacert_pem_top_level_and_vendored_are_allowed():
    for path in (
        "usr/local/lib/python3.11/site-packages/certifi/cacert.pem",
        "usr/local/lib/python3.11/site-packages/pip/_vendor/certifi/cacert.pem",
    ):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo(name=path)
            info.size = len(_REAL_CA_CERT_PEM)
            tf.addfile(info, io.BytesIO(_REAL_CA_CERT_PEM))
        buf.seek(0)
        with tarfile.open(fileobj=buf) as tf:
            member = tf.getmembers()[0]
            hit = scanner.classify_member(
                member.name, lambda tf=tf, member=member: _read_member_content(tf, member.name)
            )
        assert hit is None, path


def test_symlink_at_trusted_ca_path_is_still_credential():
    """CR-I3-MAJ-02: a symlink at a trust-store-shaped path must not be
    allowlisted by path alone — the link entry itself carries no verified
    content, so it stays `credential` regardless of its target."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="usr/lib/ssl/cert.pem")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/ssl/certs/ca-certificates.crt"
        tf.addfile(info)
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        assert member.issym()
        hit = scanner.classify_member(
            member.name, None, is_regular_file=member.isfile()
        )
    assert hit == ("credential", ".pem")


def test_malicious_private_key_under_trust_store_path_is_still_credential():
    """A private key smuggled at a trusted CA path must not be exempted —
    the allowlist checks content, not just location."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="etc/ssl/certs/evil.pem")
        info.size = len(_FAKE_PRIVATE_KEY_PEM)
        tf.addfile(info, io.BytesIO(_FAKE_PRIVATE_KEY_PEM))
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        hit = scanner.classify_member(
            member.name, lambda: _read_member_content(tf, member.name)
        )
    assert hit == ("credential", ".pem")


def test_malformed_cert_under_trust_store_path_is_still_credential():
    """CERTIFICATE-labeled but structurally invalid PEM at a trusted path
    fails ssl verification and stays fail-closed as credential."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="etc/ssl/certs/broken.pem")
        info.size = len(_MALFORMED_CERT_PEM)
        tf.addfile(info, io.BytesIO(_MALFORMED_CERT_PEM))
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        hit = scanner.classify_member(
            member.name, lambda: _read_member_content(tf, member.name)
        )
    assert hit == ("credential", ".pem")


def test_real_ca_content_outside_trust_store_path_is_still_credential():
    """Arbitrary app PEM credentials: a legitimate-looking CA cert dropped
    at a non-trust-store app path (e.g. bundled by the app itself) must
    still be flagged — the allowlist is path-scoped, not content-only."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="app/secrets/bundled_ca.pem")
        info.size = len(_REAL_CA_CERT_PEM)
        tf.addfile(info, io.BytesIO(_REAL_CA_CERT_PEM))
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        hit = scanner.classify_member(
            member.name, lambda: _read_member_content(tf, member.name)
        )
    assert hit == ("credential", ".pem")


def test_private_key_pem_outside_trust_store_path_is_credential():
    layer = _make_layer_tar(["app/keys/server.pem"])
    member = layer.getmembers()[0]
    hit = scanner.classify_member(member.name)
    assert hit == ("credential", ".pem")


def test_env_secret_file_is_still_detected():
    layer = _make_layer_tar(["app/.env.production"])
    member = layer.getmembers()[0]
    hit = scanner.classify_member(member.name)
    assert hit == ("env_file", ".env")


def test_pfx_credential_is_still_detected():
    layer = _make_layer_tar(["app/client_identity.pfx"])
    member = layer.getmembers()[0]
    hit = scanner.classify_member(member.name)
    assert hit == ("credential", ".pfx")


def test_is_verified_ca_bundle_rejects_mixed_cert_and_private_key():
    """A file concatenating a real cert with a private key block must not
    verify — every PEM label in the file must be CERTIFICATE."""
    mixed = _REAL_CA_CERT_PEM + _FAKE_PRIVATE_KEY_PEM
    assert scanner.is_verified_ca_bundle(mixed) is False


def test_is_verified_ca_bundle_rejects_content_with_no_pem_blocks():
    assert scanner.is_verified_ca_bundle(b"not a pem file at all") is False


def test_is_verified_ca_bundle_accepts_multi_cert_bundle():
    bundle = _REAL_CA_CERT_PEM + _REAL_CA_CERT_PEM
    assert scanner.is_verified_ca_bundle(bundle) is True


def test_deleted_credential_still_detected_in_earlier_layer():
    """Deletion-history leakage: OCI layers are additive, so `rm`ing a
    secret in a later layer only whites it out of the final view — the
    secret's bytes are still present, and still classified, in the layer
    that wrote them. This mirrors the per-layer loop in scan()."""
    layer_1 = _make_layer_tar(["app/secrets/id_rsa"])
    layer_2 = _make_layer_tar(["app/secrets/.wh.id_rsa"])

    all_violations = []
    for layer in (layer_1, layer_2):
        for member in layer.getmembers():
            if scanner.is_whiteout(member.name):
                continue
            hit = scanner.classify_member(member.name)
            if hit:
                all_violations.append(hit)

    assert all_violations == [("credential", "id_rsa")]


def test_duplicate_credential_in_every_layer_is_independently_detected():
    """No cross-layer dedup: the same secret path written identically in
    two separate layers (e.g. a rebuild that recreates the same file) must
    be flagged once per layer it actually appears in, never suppressed
    because an earlier layer already reported it."""
    layer_1 = _make_layer_tar(["app/secrets/id_rsa"])
    layer_2 = _make_layer_tar(["app/secrets/id_rsa"])

    all_violations = []
    for layer in (layer_1, layer_2):
        for member in layer.getmembers():
            if scanner.is_whiteout(member.name):
                continue
            hit = scanner.classify_member(member.name)
            if hit:
                all_violations.append(hit)

    assert all_violations == [("credential", "id_rsa"), ("credential", "id_rsa")]


def test_duplicate_whiteout_history_all_still_fail_closed():
    """Duplicate opaque-whiteout markers across multiple layers (e.g. a
    directory recreated and re-deleted) must never themselves be treated
    as content, and must never suppress detection of a real secret that
    precedes them."""
    layer_1 = _make_layer_tar(["app/secrets/id_rsa"])
    layer_2 = _make_layer_tar(["app/secrets/.wh..wh..opq"])
    layer_3 = _make_layer_tar(["app/secrets/.wh..wh..opq"])

    all_violations = []
    for layer in (layer_1, layer_2, layer_3):
        for member in layer.getmembers():
            if scanner.is_whiteout(member.name):
                continue
            hit = scanner.classify_member(member.name)
            if hit:
                all_violations.append(hit)

    assert all_violations == [("credential", "id_rsa")]


# --- CR-I3-MAJ-01: strict full-consumption PEM parsing ---------------------
#
# Code_Review_Iteration_3.md found is_verified_ca_bundle() accepted a real
# certificate followed by arbitrary bytes because it only scanned for BEGIN
# labels and delegated the whole string to SSLContext, which tolerates
# trailing/leading junk. The strict grammar below must reject every byte
# that isn't part of a complete BEGIN/END CERTIFICATE block or the
# whitespace between blocks.

def test_is_verified_ca_bundle_rejects_appended_secret_after_valid_cert():
    data = _REAL_CA_CERT_PEM + b"API_TOKEN=supersecret\n"
    assert scanner.is_verified_ca_bundle(data) is False


def test_is_verified_ca_bundle_rejects_prepended_secret_before_valid_cert():
    data = b"API_TOKEN=supersecret\n" + _REAL_CA_CERT_PEM
    assert scanner.is_verified_ca_bundle(data) is False


def test_is_verified_ca_bundle_rejects_secret_interleaved_between_certs():
    data = _REAL_CA_CERT_PEM + b"API_TOKEN=supersecret\n" + _REAL_CA_CERT_PEM
    assert scanner.is_verified_ca_bundle(data) is False


def test_is_verified_ca_bundle_rejects_valid_cert_plus_unlabeled_private_key_tail():
    """A valid certificate followed by a `-----END PRIVATE KEY-----` marker
    with no matching BEGIN line — the exact bypass probe from
    CR-I3-MAJ-01 — must not verify."""
    data = _REAL_CA_CERT_PEM + b"-----END PRIVATE KEY-----\n"
    assert scanner.is_verified_ca_bundle(data) is False


def test_is_verified_ca_bundle_rejects_unmatched_begin_with_no_end():
    data = b"-----BEGIN CERTIFICATE-----\nMIIEkTCCA3mgAwIBAgIERWtQVDAN\n"
    assert scanner.is_verified_ca_bundle(data) is False


def test_is_verified_ca_bundle_rejects_mismatched_end_label():
    body = _REAL_CA_CERT_PEM.split(b"\n")[1:-2]
    data = b"-----BEGIN CERTIFICATE-----\n" + b"\n".join(body) + b"\n-----END PRIVATE KEY-----\n"
    assert scanner.is_verified_ca_bundle(data) is False


def test_is_verified_ca_bundle_rejects_non_ascii_bytes():
    assert scanner.is_verified_ca_bundle(_REAL_CA_CERT_PEM + b"\xff\xfe") is False


def test_classify_member_flags_trusted_path_file_with_appended_secret():
    """End-to-end through classify_member (not just the unit-level parser):
    a regular file at a trust-store path with a valid cert plus appended
    secret bytes must still classify as credential."""
    data = _REAL_CA_CERT_PEM + b"API_TOKEN=supersecret\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="etc/ssl/certs/tampered.pem")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        hit = scanner.classify_member(
            member.name, lambda: _read_member_content(tf, member.name)
        )
    assert hit == ("credential", ".pem")


# --- CR-I3-MAJ-02: symlink/hardlink/device/FIFO must never be allowlisted -

def _add_link_member(tf: tarfile.TarFile, name: str, linkname: str, link_type: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.type = link_type
    info.linkname = linkname
    tf.addfile(info)


def test_hardlink_at_trusted_ca_path_is_still_credential():
    """A hardlink named like an allowlisted CA bundle, pointing at an
    arbitrary in-layer secret, must not be exempted by path alone."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"-----BEGIN RSA PRIVATE KEY-----\nsecret\n-----END RSA PRIVATE KEY-----\n"
        info = tarfile.TarInfo(name="app/secrets/key.pem")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        _add_link_member(
            tf, "etc/ssl/certs/innocent.pem", "app/secrets/key.pem", tarfile.LNKTYPE
        )
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        members = {m.name: m for m in tf.getmembers()}
        link = members["etc/ssl/certs/innocent.pem"]
        assert link.islnk()
        hit = scanner.classify_member(
            link.name, None, is_regular_file=link.isfile()
        )
    assert hit == ("credential", ".pem")


def test_symlink_pointing_at_arbitrary_secret_is_still_credential():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        _add_link_member(
            tf, "etc/ssl/certs/innocent.pem", "/app/secrets/key.pem", tarfile.SYMTYPE
        )
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        assert member.issym()
        hit = scanner.classify_member(
            member.name, None, is_regular_file=member.isfile()
        )
    assert hit == ("credential", ".pem")


def test_symlink_target_traversal_does_not_grant_allowlist():
    """A symlink at a trust-store path whose target attempts to traverse
    out of the layer must still be denied — classification never follows
    or resolves the traversal target, so it cannot be used to smuggle a
    pass."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        _add_link_member(
            tf,
            "etc/ssl/certs/innocent.pem",
            "../../../../etc/shadow",
            tarfile.SYMTYPE,
        )
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        hit = scanner.classify_member(
            member.name, None, is_regular_file=member.isfile()
        )
    assert hit == ("credential", ".pem")


def test_hardlink_target_traversal_does_not_grant_allowlist():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        _add_link_member(
            tf,
            "etc/ssl/certs/innocent.pem",
            "../../../../app/secrets/id_rsa",
            tarfile.LNKTYPE,
        )
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        hit = scanner.classify_member(
            member.name, None, is_regular_file=member.isfile()
        )
    assert hit == ("credential", ".pem")


def test_character_device_at_trusted_ca_path_is_still_credential():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="etc/ssl/certs/device.pem")
        info.type = tarfile.CHRTYPE
        info.devmajor = 1
        info.devminor = 5
        tf.addfile(info)
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        assert member.ischr()
        hit = scanner.classify_member(
            member.name, None, is_regular_file=member.isfile()
        )
    assert hit == ("credential", ".pem")


def test_fifo_at_trusted_ca_path_is_still_credential():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="etc/ssl/certs/pipe.pem")
        info.type = tarfile.FIFOTYPE
        tf.addfile(info)
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        assert member.isfifo()
        hit = scanner.classify_member(
            member.name, None, is_regular_file=member.isfile()
        )
    assert hit == ("credential", ".pem")


def test_directory_at_trusted_ca_path_is_not_regular_file_and_stays_denied():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="etc/ssl/certs/cert.pem")
        info.type = tarfile.DIRTYPE
        tf.addfile(info)
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tf:
        member = tf.getmembers()[0]
        assert member.isdir()
        hit = scanner.classify_member(
            member.name, None, is_regular_file=member.isfile()
        )
    assert hit == ("credential", ".pem")


def test_trusted_ca_crt_extension_is_flagged_when_untrusted():
    """The generic forbidden-pattern list must recognize `.crt`, not just
    `.pem` — otherwise a `.crt` file outside the allowlist (or a non-regular
    member at a trusted `.crt` path) silently passes with no pattern able
    to catch it."""
    layer = _make_layer_tar(["app/secrets/bundled_ca.crt"])
    member = layer.getmembers()[0]
    hit = scanner.classify_member(member.name)
    assert hit == ("credential", ".crt")


def test_scan_flags_hardlink_bypass_end_to_end(monkeypatch, tmp_path):
    """Full scan() path (not just classify_member directly): a layer
    containing a real secret plus an allowlisted-looking hardlink to it
    must surface as a violation, proving the fix closes the gap the
    reviewer reproduced against scan()'s actual call site.

    CR-I4-MIN-01: the oracle asserts the *exact* violation record for the
    hardlink member itself (not just "some credential exists somewhere"),
    so this test cannot pass if the hardlink is mistakenly exempted while
    some unrelated fixture member happens to also be a credential."""
    layer_buf = io.BytesIO()
    with tarfile.open(fileobj=layer_buf, mode="w") as tf:
        data = b"super-secret-key-bytes"
        info = tarfile.TarInfo(name="app/secrets/key.pem")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        _add_link_member(
            tf, "etc/ssl/certs/innocent.pem", "app/secrets/key.pem", tarfile.LNKTYPE
        )
    layer_bytes = layer_buf.getvalue()

    image_buf = io.BytesIO()
    with tarfile.open(fileobj=image_buf, mode="w") as outer:
        layer_info = tarfile.TarInfo(name="layer.tar")
        layer_info.size = len(layer_bytes)
        outer.addfile(layer_info, io.BytesIO(layer_bytes))

        manifest = json.dumps([{"Layers": ["layer.tar"]}]).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="manifest.json")
        manifest_info.size = len(manifest)
        outer.addfile(manifest_info, io.BytesIO(manifest))
    image_bytes = image_buf.getvalue()

    archive_path = tmp_path / "image.tar"
    archive_path.write_bytes(image_bytes)

    def fake_export_image(image: str, out_tar):
        out_tar.write_bytes(image_bytes)

    monkeypatch.setattr(scanner, "export_image", fake_export_image)
    result = scanner.scan("fake:image")
    by_member = {v["member"]: v for v in result["violations"]}
    assert by_member["etc/ssl/certs/innocent.pem"] == {
        "layer": "layer.tar",
        "member": "etc/ssl/certs/innocent.pem",
        "category": "credential",
        "pattern": ".pem",
    }
    assert by_member["app/secrets/key.pem"] == {
        "layer": "layer.tar",
        "member": "app/secrets/key.pem",
        "category": "credential",
        "pattern": ".pem",
    }
    assert result["forbidden_count"] == 2


# --- Hosted-CI remediation iteration 3: real Debian/certifi shapes --------
#
# Code_Review_Iteration_4.md's hosted run 31609022196 showed the
# iteration-2 conservative link rejection produces false positives against
# the *real* Debian base image (docker run inspection of the built
# `production` target confirmed the shapes below), and that the
# iteration-2 strict grammar rejects the *real* certifi cacert.pem, whose
# upstream format interleaves fixed comment lines before each certificate.
# Everything below proves the iteration-3 fix accepts these exact real
# shapes while staying fail-closed against dangling targets, cycles,
# traversal, whiteout ambiguity, and comment-disguised secrets.

# Exact real comment stanza for the same Entrust root as _REAL_CA_CERT_PEM,
# copied verbatim from the installed certifi package's cacert.pem (all
# seven fields, correct order, byte-exact fingerprint lengths) — Code
# Review Iteration 5 (CR-I5-MAJ-01) requires the grammar to enforce exact
# field count/order/length, so this fixture must be genuine, not a
# hand-abbreviated approximation (a shortened, e.g. non-32-byte, SHA256
# value would now correctly fail to match).
_CERTIFI_STYLE_BLOCK = (
    b"# Issuer: CN=Entrust Root Certification Authority O=Entrust, Inc. "
    b"OU=www.entrust.net/CPS is incorporated by reference/(c) 2006 Entrust, Inc.\n"
    b"# Subject: CN=Entrust Root Certification Authority O=Entrust, Inc. "
    b"OU=www.entrust.net/CPS is incorporated by reference/(c) 2006 Entrust, Inc.\n"
    b'# Label: "Entrust Root Certification Authority"\n'
    b"# Serial: 1164660820\n"
    b"# MD5 Fingerprint: d6:a5:c3:ed:5d:dd:3e:00:c1:3d:87:92:1f:1d:3f:e4\n"
    b"# SHA1 Fingerprint: b3:1e:b1:b7:40:e3:6c:84:02:da:dc:37:d4:4d:f5:d4:67:49:52:f9\n"
    b"# SHA256 Fingerprint: 73:c1:76:43:4f:1b:c6:d5:ad:f4:5b:0e:76:e7:27:28:"
    b"7c:8d:e5:76:16:c1:e6:e6:14:1a:2b:2c:bc:7d:8e:4c\n"
) + _REAL_CA_CERT_PEM


def test_is_verified_ca_bundle_accepts_real_certifi_comment_format():
    """The exact upstream certifi shape (fixed metadata comments
    immediately before each CERTIFICATE block, blank line between
    entries) must verify — this is the iteration-2 regression that broke
    the real pip-vendored and top-level certifi/cacert.pem."""
    bundle = _CERTIFI_STYLE_BLOCK + b"\n" + _CERTIFI_STYLE_BLOCK
    assert scanner.is_verified_ca_bundle(bundle) is True


def test_is_verified_ca_bundle_rejects_unrecognized_comment_field():
    """A comment that isn't one of the seven recognized certifi field
    prefixes (e.g. a smuggled note or a disguised secret) must still break
    full consumption — the comment allowance is narrow, not general."""
    tampered = _CERTIFI_STYLE_BLOCK.replace(b"# Label:", b"# NotARealField:")
    assert scanner.is_verified_ca_bundle(tampered) is False


def test_is_verified_ca_bundle_rejects_secret_disguised_as_comment():
    data = b"# API_TOKEN=supersecret-not-a-real-field\n" + _REAL_CA_CERT_PEM
    assert scanner.is_verified_ca_bundle(data) is False


def test_is_verified_ca_bundle_rejects_comment_after_cert_block():
    """Recognized comment lines are only permitted immediately *before* a
    BEGIN CERTIFICATE block, matching the real certifi layout — a comment
    trailing a block (not leading the next one) still fails full
    consumption."""
    data = _REAL_CA_CERT_PEM + b"# Label: \"trailing, not leading\"\n"
    assert scanner.is_verified_ca_bundle(data) is False


# --- Code Review Iteration 5 (CR-I5-MAJ-01): exact certifi stanza grammar -
#
# The iteration-3 grammar (`(?:_COMMENT_LINE)*` with a `[^\r\n]{0,512}`
# value class per recognized prefix) accepted any number and order of the
# seven prefixes with unbounded free-text values — direct adversarial
# probes with duplicated/reordered/token-bearing "recognized-prefix"
# comments all still matched (Code_Review_Iteration_5.md CR-I5-MAJ-01).
# The fix encodes the seven fields as one fixed, non-repeating, exactly
# once each sequence with field-specific bounded grammar (see the module
# docstring). Everything below proves: a full real installed certifi
# bundle still verifies end to end; missing/duplicate/reordered/extra
# fields are all rejected; and every one of the seven fields rejects
# token/path/key-value/private-material payloads under its own grammar.

_CERTIFI_ISSUER_SUBJECT = (
    b"CN=Entrust Root Certification Authority O=Entrust, Inc. "
    b"OU=www.entrust.net/CPS is incorporated by reference/(c) 2006 Entrust, Inc."
)

# Exact real entry copied from the real linux/amd64 production image's
# `pip/_vendor/certifi/cacert.pem` (docker cp against a container built
# from this exact revision's deploy/Dockerfile). Verifying this exact
# byte-for-byte real stanza is what actually caught the CR-I5-MAJ-01
# remediation's own regression: the first Issuer/Subject-only alphabet
# (no underscore) rejected this genuine, untampered entry — pip's
# vendored certifi renders `OU=...CPS_2048...` with a literal underscore
# — producing a real `forbidden_count: 1` false positive against the
# actual built image. `_ISSUER_SUBJECT_VALUE` now includes underscore.
_REAL_PIP_VENDORED_CERTIFI_ENTRUST_2048_ENTRY = (
    b"# Issuer: CN=Entrust.net Certification Authority (2048) O=Entrust.net "
    b"OU=www.entrust.net/CPS_2048 incorp. by ref. (limits liab.)/(c) 1999 "
    b"Entrust.net Limited\n"
    b"# Subject: CN=Entrust.net Certification Authority (2048) O=Entrust.net "
    b"OU=www.entrust.net/CPS_2048 incorp. by ref. (limits liab.)/(c) 1999 "
    b"Entrust.net Limited\n"
    b'# Label: "Entrust.net Premium 2048 Secure Server CA"\n'
    b"# Serial: 946069240\n"
    b"# MD5 Fingerprint: ee:29:31:bc:32:7e:9a:e6:e8:b5:f7:51:b4:34:71:90\n"
    b"# SHA1 Fingerprint: 50:30:06:09:1d:97:d4:f5:ae:39:f7:cb:e7:92:7d:7d:65:2d:34:31\n"
    b"# SHA256 Fingerprint: 6d:c4:71:72:e0:1c:bc:b0:bf:62:58:0d:89:5f:e2:b8:"
    b"ac:9a:d4:f8:73:80:1e:0c:10:b9:c8:37:d2:1e:b1:77\n"
    b"-----BEGIN CERTIFICATE-----\n"
    b"MIIEKjCCAxKgAwIBAgIEOGPe+DANBgkqhkiG9w0BAQUFADCBtDEUMBIGA1UEChML\n"
    b"RW50cnVzdC5uZXQxQDA+BgNVBAsUN3d3dy5lbnRydXN0Lm5ldC9DUFNfMjA0OCBp\n"
    b"bmNvcnAuIGJ5IHJlZi4gKGxpbWl0cyBsaWFiLikxJTAjBgNVBAsTHChjKSAxOTk5\n"
    b"IEVudHJ1c3QubmV0IExpbWl0ZWQxMzAxBgNVBAMTKkVudHJ1c3QubmV0IENlcnRp\n"
    b"ZmljYXRpb24gQXV0aG9yaXR5ICgyMDQ4KTAeFw05OTEyMjQxNzUwNTFaFw0yOTA3\n"
    b"MjQxNDE1MTJaMIG0MRQwEgYDVQQKEwtFbnRydXN0Lm5ldDFAMD4GA1UECxQ3d3d3\n"
    b"LmVudHJ1c3QubmV0L0NQU18yMDQ4IGluY29ycC4gYnkgcmVmLiAobGltaXRzIGxp\n"
    b"YWIuKTElMCMGA1UECxMcKGMpIDE5OTkgRW50cnVzdC5uZXQgTGltaXRlZDEzMDEG\n"
    b"A1UEAxMqRW50cnVzdC5uZXQgQ2VydGlmaWNhdGlvbiBBdXRob3JpdHkgKDIwNDgp\n"
    b"MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArU1LqRKGsuqjIAcVFmQq\n"
    b"K0vRvwtKTY7tgHalZ7d4QMBzQshowNtTK91euHaYNZOLGp18EzoOH1u3Hs/lJBQe\n"
    b"sYGpjX24zGtLA/ECDNyrpUAkAH90lKGdCCmziAv1h3edVc3kw37XamSrhRSGlVuX\n"
    b"MlBvPci6Zgzj/L24ScF2iUkZ/cCovYmjZy/Gn7xxGWC4LeksyZB2ZnuU4q941mVT\n"
    b"XTzWnLLPKQP5L6RQstRIzgUyVYr9smRMDuSYB3Xbf9+5CFVghTAp+XtIpGmG4zU/\n"
    b"HoZdenoVve8AjhUiVBcAkCaTvA5JaJG/+EfTnZVCwQ5N328mz8MYIWJmQ3DW1cAH\n"
    b"4QIDAQABo0IwQDAOBgNVHQ8BAf8EBAMCAQYwDwYDVR0TAQH/BAUwAwEB/zAdBgNV\n"
    b"HQ4EFgQUVeSB0RGAvtiJuQijMfmhJAkWuXAwDQYJKoZIhvcNAQEFBQADggEBADub\n"
    b"j1abMOdTmXx6eadNl9cZlZD7Bh/KM3xGY4+WZiT6QBshJ8rmcnPyT/4xmf3IDExo\n"
    b"U8aAghOY+rat2l098c5u9hURlIIM7j+VrxGrD9cv3h8Dj1csHsm7mhpElesYT6Yf\n"
    b"zX1XEC+bBAlahLVu2B064dae0Wx5XnkcFMXj0EyTO2U87d89vqbllRrDtRnDvV5b\n"
    b"u/8j72gZyxKTJ1wDLW8w0B62GqzeWvfRqqgnpv55gcR5mTNXuhKwqeBCbJPKVt7+\n"
    b"bYQLCIt+jerXmCHG8+c8eS9enNFMFY3h7CI3zJpDC5fcgJCNs2ebb0gIFVbPv/Er\n"
    b"fF6adulZkMV8gzURZVE=\n"
    b"-----END CERTIFICATE-----\n"
)


def test_is_verified_ca_bundle_accepts_real_pip_vendored_entry_with_underscore_in_dn():
    """CR-I5-MAJ-01 remediation regression: a genuine DN legitimately
    contains an underscore (`OU=...CPS_2048...`), and the Issuer/Subject
    alphabet must accept it — an alphabet narrow enough to exclude
    underscore entirely would false-positive-reject this real,
    untampered certifi entry (reproduced against the real built image;
    see scan_image_layers.py's `_ISSUER_SUBJECT_VALUE` docstring)."""
    assert (
        scanner.is_verified_ca_bundle(_REAL_PIP_VENDORED_CERTIFI_ENTRUST_2048_ENTRY)
        is True
    )


def test_pip_vendored_entrust_label_exception_is_certificate_bound():
    tampered = _REAL_PIP_VENDORED_CERTIFI_ENTRUST_2048_ENTRY.replace(
        b'"Entrust.net Premium 2048 Secure Server CA"',
        b'"Entrust.net Premium 2048 Secure Server CB"',
        1,
    )
    assert scanner.is_verified_ca_bundle(tampered) is False


def _certifi_stanza(
    *,
    issuer: bytes = _CERTIFI_ISSUER_SUBJECT,
    subject: bytes = _CERTIFI_ISSUER_SUBJECT,
    label: bytes = b'"Entrust Root Certification Authority"',
    serial: bytes = b"1164660820",
    md5: bytes = b"d6:a5:c3:ed:5d:dd:3e:00:c1:3d:87:92:1f:1d:3f:e4",
    sha1: bytes = b"b3:1e:b1:b7:40:e3:6c:84:02:da:dc:37:d4:4d:f5:d4:67:49:52:f9",
    sha256: bytes = (
        b"73:c1:76:43:4f:1b:c6:d5:ad:f4:5b:0e:76:e7:27:28:"
        b"7c:8d:e5:76:16:c1:e6:e6:14:1a:2b:2c:bc:7d:8e:4c"
    ),
) -> bytes:
    """Build a certifi-shaped stanza + the real Entrust cert, with one
    field's value swappable for adversarial probing. Defaults reproduce
    the exact real upstream values so only the targeted field deviates."""
    return (
        b"# Issuer: " + issuer + b"\n"
        b"# Subject: " + subject + b"\n"
        b"# Label: " + label + b"\n"
        b"# Serial: " + serial + b"\n"
        b"# MD5 Fingerprint: " + md5 + b"\n"
        b"# SHA1 Fingerprint: " + sha1 + b"\n"
        b"# SHA256 Fingerprint: " + sha256 + b"\n"
    ) + _REAL_CA_CERT_PEM


def test_is_verified_ca_bundle_accepts_full_installed_certifi_bundle():
    """Full real upstream oracle: the actual installed `certifi` package
    (a locked project dependency — see requirements.lock) exports its
    entire real `cacert.pem`, comments and all, and it must verify in one
    shot — not a hand-built approximation of the format."""
    import certifi

    with open(certifi.where(), "rb") as handle:
        data = handle.read()
    assert len(data) > 0
    assert scanner.is_verified_ca_bundle(data) is True


def test_is_verified_ca_bundle_accepts_full_pip_vendored_certifi_bundle():
    """Same full-bundle oracle as above, against pip's independently
    vendored `certifi` copy (`pip/_vendor/certifi/cacert.pem`) rather than
    the top-level package — CR-I7-MAJ-01 remediation: the switch from
    case/whitespace-insensitive Label matching to exact equality plus a
    fixed per-certificate exception table must not regress either real,
    unmodified installed bundle."""
    from pip._vendor import certifi as pip_certifi

    with open(pip_certifi.where(), "rb") as handle:
        data = handle.read()
    assert len(data) > 0
    assert scanner.is_verified_ca_bundle(data) is True


# --- Code Review Iteration 8 (CR-I8-MAJ-01): documented compatibility -----
# boundary + five legacy repository-default-Python exceptions -------------
#
# The Iteration 7 remediation's three-entry exception table was verified
# against one machine's installed bundles and happened to cover every
# deviation in them, but its report incorrectly described the pip-vendored
# bundle's version as `certifi==2026.7.22` — that version only ever applied
# to the top-level `certifi` package. pip's vendored copy
# (`pip/_vendor/certifi/cacert.pem`) is baked into the installed `pip`
# package itself, is never touched by installing or upgrading the
# top-level package, and instead tracks whatever `pip` release is actually
# on `PATH` — so it legitimately differs between this project's `venv`
# (created with a newer `pip`) and this machine's repository-default
# Python (the interpreter Code_Review_Iteration_8.md's fresh review
# actually ran under: `python3`/`pytest` resolved without activating
# `venv`, whose `pip==23.3.1` vendors `certifi==2023.07.22`). See the
# module docstring for the full three-boundary write-up.
#
# Walking that real, unmodified certifi==2023.07.22 bundle (independently,
# not just trusting the review's list) finds five ordinary-Label
# deviations beyond the three already-covered entries: Comodo AAA Services
# root, Security Communication Root CA, XRamp Global CA Root, Go Daddy
# Class 2 CA, and Starfield Class 2 CA. Each is added to
# `_CERTIFICATE_LABEL_COMPATIBILITY` as its own exact (certificate SHA-256,
# exact Label) pair. Rather than embedding all five real certificates'
# full PEM bytes here (a large, hard-to-review diff for data the existing
# full pip-vendored bundle test already exercises end-to-end when run
# under the repository-default interpreter — see
# test_is_verified_ca_bundle_accepts_full_pip_vendored_certifi_bundle and
# Code_Review_Iteration_8_Remediation.md's verification section), the
# tests below call `_label_bound_to_subject` directly with the exact real
# SHA-256/Label pair plus a genuine decoded Subject RDN value (taken from
# the real certificate, but expressed as the same `((attr_type, value),
# ...)` tuple shape `ssl._ssl._test_decode_cert` returns, not a hand-built
# guess) that differs from the Label — proving each exception both accepts
# the real deviation and stays certificate-bound (a mutated Label or wrong
# SHA-256 on the same Subject must still fail).

_CERTIFI_2023_LEGACY_LABEL_EXCEPTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "comodo_aaa_services_root",
        "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4",
        "Comodo AAA Services root",
        "Comodo CA Limited",
    ),
    (
        "security_communication_root_ca",
        "e75e72ed9f560eec6eb4800073a43fc3ad19195a392282017895974a99026b6c",
        "Security Communication Root CA",
        "SECOM Trust.net",
    ),
    (
        "xramp_global_ca_root",
        "cecddc905099d8dadfc5b1d209b737cbe2c18cfb2c10c0ff0bcf0d3286fc1aa2",
        "XRamp Global CA Root",
        "XRamp Security Services Inc",
    ),
    (
        "go_daddy_class_2_ca",
        "c3846bf24b9e93ca64274c0ec67c1ecc5e024ffcacd2d74019350e81fe546ae4",
        "Go Daddy Class 2 CA",
        "The Go Daddy Group, Inc.",
    ),
    (
        "starfield_class_2_ca",
        "1465fa205397b876faa6f0a9958e5590e40fcc7faa4fb7c2c8677521fb5fb658",
        "Starfield Class 2 CA",
        "Starfield Technologies, Inc.",
    ),
)


def _single_rdn_subject(organization_value: str) -> tuple:
    return ((("organizationName", organization_value),),)


@pytest.mark.parametrize(
    "name, sha256, label, organization_value",
    _CERTIFI_2023_LEGACY_LABEL_EXCEPTIONS,
    ids=[entry[0] for entry in _CERTIFI_2023_LEGACY_LABEL_EXCEPTIONS],
)
def test_label_bound_to_subject_accepts_certifi_2023_legacy_exception(
    name, sha256, label, organization_value
):
    """Each real certifi==2023.07.22 deviation the fresh review reported
    binds via its exact table entry even though the Label isn't any
    decoded Subject RDN value."""
    subject_rdns = _single_rdn_subject(organization_value)
    assert label not in {value for rdn in subject_rdns for _, value in rdn}
    assert scanner._label_bound_to_subject(label, subject_rdns, sha256) is True


@pytest.mark.parametrize(
    "name, sha256, label, organization_value",
    _CERTIFI_2023_LEGACY_LABEL_EXCEPTIONS,
    ids=[entry[0] for entry in _CERTIFI_2023_LEGACY_LABEL_EXCEPTIONS],
)
def test_label_bound_to_subject_rejects_mutated_legacy_label(
    name, sha256, label, organization_value
):
    """The exception is bound to one exact Label string, not any Label for
    that certificate's SHA-256 — a one-character mutation must fail."""
    subject_rdns = _single_rdn_subject(organization_value)
    mutated_label = label + "X"
    assert scanner._label_bound_to_subject(mutated_label, subject_rdns, sha256) is False


@pytest.mark.parametrize(
    "name, sha256, label, organization_value",
    _CERTIFI_2023_LEGACY_LABEL_EXCEPTIONS,
    ids=[entry[0] for entry in _CERTIFI_2023_LEGACY_LABEL_EXCEPTIONS],
)
def test_label_bound_to_subject_rejects_legacy_label_with_wrong_sha256(
    name, sha256, label, organization_value
):
    """The exception is bound to one exact certificate SHA-256, not any
    certificate that happens to carry this exact Label text."""
    subject_rdns = _single_rdn_subject(organization_value)
    wrong_sha256 = "0" * 64
    assert wrong_sha256 != sha256
    assert scanner._label_bound_to_subject(label, subject_rdns, wrong_sha256) is False


# --- Code Review Iteration 7 (CR-I7-MAJ-01): exact ordinary Label binding --
#
# The prior `_label_bound_to_subject` compared
# `label_value.strip().casefold()` against case-folded Subject RDN
# candidates, so a case-only or leading/trailing-whitespace mutation of a
# genuine Label still bound to the certificate — the seventh field was the
# sole non-exact one of the seven. These probes are the exact ones the
# review demonstrated against the real Entrust Root Certification Authority
# fixture (`_certifi_stanza`'s default label): each must now be rejected,
# while the unmodified genuine label (already covered by
# `test_is_verified_ca_bundle_accepts_real_certifi_comment_format` and the
# full-bundle oracles above) continues to pass.


@pytest.mark.parametrize(
    "label",
    [
        b'"entrust Root Certification Authority"',
        b'" Entrust Root Certification Authority"',
        b'"Entrust Root Certification Authority "',
    ],
)
def test_is_verified_ca_bundle_rejects_case_or_whitespace_label_variant(label):
    """Case-only and leading/trailing-space variants of a genuine Label,
    against the otherwise exact real certificate and the other six exact
    fields, must fail — Label binding is byte-for-byte exact, not a
    case/whitespace equivalence class."""
    assert scanner.is_verified_ca_bundle(_certifi_stanza(label=label)) is False


@pytest.mark.parametrize(
    "prefix",
    [
        b"# Issuer:",
        b"# Subject:",
        b"# Label:",
        b"# Serial:",
        b"# MD5 Fingerprint:",
        b"# SHA1 Fingerprint:",
        b"# SHA256 Fingerprint:",
    ],
)
def test_is_verified_ca_bundle_rejects_stanza_missing_a_field(prefix):
    """Dropping any single one of the seven fields must break the fixed
    sequence — six of seven is not a valid stanza."""
    lines = _CERTIFI_STYLE_BLOCK.split(b"\n")
    dropped = b"\n".join(line for line in lines if not line.startswith(prefix))
    assert scanner.is_verified_ca_bundle(dropped) is False, prefix


def test_is_verified_ca_bundle_rejects_duplicate_field():
    """A duplicated `# Issuer:` line ahead of the real one (with `#
    Subject:` still following where the grammar expects it) must not be
    tolerated as an eight-line variant of the seven-field stanza."""
    duplicated = _CERTIFI_STYLE_BLOCK.replace(
        b"# Issuer: CN=Entrust",
        b"# Issuer: CN=Entrust\n# Issuer: CN=Entrust",
        1,
    )
    assert scanner.is_verified_ca_bundle(duplicated) is False


def test_is_verified_ca_bundle_rejects_reordered_fields():
    """The seven fields must appear in the exact upstream order — swapping
    `# Label:` and `# Serial:` must break the match even though all seven
    recognized prefixes and legal values are still present somewhere in
    the stanza."""
    lines = _CERTIFI_STYLE_BLOCK.split(b"\n")
    label_idx = next(i for i, line in enumerate(lines) if line.startswith(b"# Label:"))
    serial_idx = next(i for i, line in enumerate(lines) if line.startswith(b"# Serial:"))
    lines[label_idx], lines[serial_idx] = lines[serial_idx], lines[label_idx]
    reordered = b"\n".join(lines)
    assert scanner.is_verified_ca_bundle(reordered) is False


def test_is_verified_ca_bundle_rejects_extra_field():
    """An eighth line — even a well-formed, recognized-prefix repeat of
    the last field — must not be tolerated alongside the required seven;
    the sequence is fixed-length, not open-ended."""
    extra = _CERTIFI_STYLE_BLOCK.replace(
        b"# Serial: 1164660820\n",
        b"# Serial: 1164660820\n# Serial: 1164660820\n",
        1,
    )
    assert scanner.is_verified_ca_bundle(extra) is False


_ADVERSARIAL_FIELD_VALUES: tuple[tuple[str, bytes], ...] = (
    # Issuer/Subject allow underscore (a real certifi DN legitimately
    # contains it — see the module docstring), so the discriminating
    # smuggled character here is the colon, which stays excluded from
    # every field's alphabet.
    ("issuer", b"token: supersecret_value_123"),
    ("subject", b"Authorization: Bearer_abc_def_ghi"),
    ("label", b'"../../etc/shadow"'),
    ("serial", b"arbitrary free text"),
    ("md5", b"sk-live-not-a-real-api-key-0123456789abcdef"),
    ("sha1", b"/etc/shadow:0:0:root:x:0:0"),
    ("sha256", b'{"token": "supersecret", "path": "/etc/shadow"}'),
)


@pytest.mark.parametrize("field, value", _ADVERSARIAL_FIELD_VALUES)
def test_is_verified_ca_bundle_rejects_token_path_key_value_material_per_field(field, value):
    """CR-I5-MAJ-01: every recognized-prefix field must reject a
    token/path/key-value/private-material payload under its own
    field-specific grammar — the prior `[^\\r\\n]{0,512}` value class
    accepted every one of these."""
    stanza = _certifi_stanza(**{field: value})
    assert scanner.is_verified_ca_bundle(stanza) is False, field


@pytest.mark.parametrize("field", ["issuer", "subject"])
@pytest.mark.parametrize(
    "value",
    [
        b"CN=API_TOKEN=supersecret",
        b"CN=../../etc/shadow",
        b"CN=PRIVATE KEY",
        b"CN=AWS_SECRET_ACCESS_KEY=ABCDEF",
    ],
)
def test_is_verified_ca_bundle_rejects_grammar_valid_dn_smuggling(field, value):
    assert scanner.is_verified_ca_bundle(_certifi_stanza(**{field: value})) is False


@pytest.mark.parametrize(
    "label",
    [
        b'"API TOKEN supersecret"',
        b'"PRIVATE KEY"',
        b'"AWS SECRET ACCESS KEY"',
        b'"etc shadow"',
    ],
)
def test_is_verified_ca_bundle_rejects_grammar_valid_label_smuggling(label):
    assert scanner.is_verified_ca_bundle(_certifi_stanza(label=label)) is False


def test_is_verified_ca_bundle_rejects_reordered_recognized_prefix_example():
    """Reproduces the exact adversarial probe from Code_Review_Iteration_5.md
    CR-I5-MAJ-01 verbatim (missing fields, reordered/duplicated Issuer,
    free-text Serial) appended before a real certificate — this is the
    literal bypass the reviewer demonstrated against the prior grammar."""
    adversarial = (
        b"# Issuer: API_TOKEN=supersecret\n"
        b"# Label: ../../etc/shadow\n"
        b"# Serial: arbitrary free text\n"
        b"# Issuer: x\n"
        b"# Issuer: y\n"
    ) + _REAL_CA_CERT_PEM
    assert scanner.is_verified_ca_bundle(adversarial) is False


def _make_image(layers: list[tuple[str, bytes]]) -> bytes:
    """Build a minimal `docker save`-shaped tar: named layer tars plus a
    manifest.json referencing them in order."""
    image_buf = io.BytesIO()
    with tarfile.open(fileobj=image_buf, mode="w") as outer:
        for name, data in layers:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            outer.addfile(info, io.BytesIO(data))
        manifest = json.dumps([{"Layers": [name for name, _ in layers]}]).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="manifest.json")
        manifest_info.size = len(manifest)
        outer.addfile(manifest_info, io.BytesIO(manifest))
    return image_buf.getvalue()


def _make_layer_bytes(entries: list[tuple[str, str, bytes | str]]) -> bytes:
    """entries: (name, kind, payload) where kind is "reg"/"sym"/"hard" and
    payload is file bytes for "reg" or a linkname string for "sym"/"hard"."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, kind, payload in entries:
            if kind == "reg":
                data = payload
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            elif kind == "sym":
                info = tarfile.TarInfo(name=name)
                info.type = tarfile.SYMTYPE
                info.linkname = payload
                tf.addfile(info)
            elif kind == "hard":
                info = tarfile.TarInfo(name=name)
                info.type = tarfile.LNKTYPE
                info.linkname = payload
                tf.addfile(info)
            else:
                raise ValueError(kind)
    return buf.getvalue()


def test_scan_allows_debian_two_hop_symlink_chain_end_to_end(monkeypatch):
    """Real Debian shape confirmed against the built production image:
    `etc/ssl/certs/<hash>.0` -> `<CN>.pem` (relative, same dir) ->
    `/usr/share/ca-certificates/mozilla/<CN>.crt` (absolute, regular file
    with genuine CA content). Both link hops must resolve and verify with
    zero violations — this is the exact false-positive hosted run
    31609022196 hit 153 times."""
    layer = _make_layer_bytes([
        (
            "usr/share/ca-certificates/mozilla/GlobalSign_Root_R46.crt",
            "reg",
            _REAL_CA_CERT_PEM,
        ),
        (
            "etc/ssl/certs/GlobalSign_Root_R46.pem",
            "sym",
            "/usr/share/ca-certificates/mozilla/GlobalSign_Root_R46.crt",
        ),
        ("etc/ssl/certs/002c0b4f.0", "sym", "GlobalSign_Root_R46.pem"),
    ])
    image_bytes = _make_image([("layer.tar", layer)])

    monkeypatch.setattr(
        scanner, "export_image", lambda image, out_tar: out_tar.write_bytes(image_bytes)
    )
    result = scanner.scan("fake:debian")
    assert result["violations"] == []
    assert result["forbidden_count"] == 0


def test_scan_allows_usr_lib_ssl_symlink_to_ca_certificates_crt(monkeypatch):
    """Real Debian shape: `/usr/lib/ssl/cert.pem` -> absolute
    `/etc/ssl/certs/ca-certificates.crt` (a genuine regular bundle)."""
    layer = _make_layer_bytes([
        ("etc/ssl/certs/ca-certificates.crt", "reg", _REAL_CA_CERT_PEM),
        ("usr/lib/ssl/cert.pem", "sym", "/etc/ssl/certs/ca-certificates.crt"),
    ])
    image_bytes = _make_image([("layer.tar", layer)])

    monkeypatch.setattr(
        scanner, "export_image", lambda image, out_tar: out_tar.write_bytes(image_bytes)
    )
    result = scanner.scan("fake:usrlibssl")
    assert result["violations"] == []


def test_scan_allows_cross_layer_symlink_resolution(monkeypatch):
    """OCI-layer-state-aware resolution: the regular CA file is written in
    an earlier layer and the symlink pointing at it is written in a later
    layer — real Docker images commonly split base-OS and later
    RUN-layer changes this way. Resolution must look across layers, not
    just within the symlink's own layer."""
    target_layer = _make_layer_bytes([
        (
            "usr/share/ca-certificates/mozilla/GlobalSign_Root_R46.crt",
            "reg",
            _REAL_CA_CERT_PEM,
        ),
    ])
    link_layer = _make_layer_bytes([
        (
            "etc/ssl/certs/GlobalSign_Root_R46.pem",
            "sym",
            "/usr/share/ca-certificates/mozilla/GlobalSign_Root_R46.crt",
        ),
    ])
    image_bytes = _make_image([("layer1.tar", target_layer), ("layer2.tar", link_layer)])

    monkeypatch.setattr(
        scanner, "export_image", lambda image, out_tar: out_tar.write_bytes(image_bytes)
    )
    result = scanner.scan("fake:crosslayer")
    assert result["violations"] == []


def test_scan_rejects_dangling_symlink_at_trusted_path(monkeypatch):
    layer = _make_layer_bytes([
        ("etc/ssl/certs/dangling.pem", "sym", "/etc/ssl/certs/does-not-exist.crt"),
    ])
    image_bytes = _make_image([("layer.tar", layer)])

    monkeypatch.setattr(
        scanner, "export_image", lambda image, out_tar: out_tar.write_bytes(image_bytes)
    )
    result = scanner.scan("fake:dangling")
    assert [v["member"] for v in result["violations"]] == ["etc/ssl/certs/dangling.pem"]


def test_scan_rejects_symlink_cycle_at_trusted_path(monkeypatch):
    layer = _make_layer_bytes([
        ("etc/ssl/certs/a.pem", "sym", "b.pem"),
        ("etc/ssl/certs/b.pem", "sym", "a.pem"),
    ])
    image_bytes = _make_image([("layer.tar", layer)])

    monkeypatch.setattr(
        scanner, "export_image", lambda image, out_tar: out_tar.write_bytes(image_bytes)
    )
    result = scanner.scan("fake:cycle")
    assert {v["member"] for v in result["violations"]} == {
        "etc/ssl/certs/a.pem",
        "etc/ssl/certs/b.pem",
    }
    assert result["forbidden_count"] == 2


def test_scan_rejects_symlink_target_whited_out_before_this_layer(monkeypatch):
    """Deletion/whiteout ambiguity: the regular CA file is written in
    layer 1, deleted (whiteout) in layer 2, and a symlink pointing at that
    now-masked path is written in layer 3 — the merged state as of layer 3
    must not still see the deleted file, so the symlink is dangling and
    stays credential."""
    layer1 = _make_layer_bytes([
        ("etc/ssl/certs/real.crt", "reg", _REAL_CA_CERT_PEM),
    ])
    layer2 = _make_layer_bytes([
        ("etc/ssl/certs/.wh.real.crt", "reg", b""),
    ])
    layer3 = _make_layer_bytes([
        ("etc/ssl/certs/pointer.pem", "sym", "real.crt"),
    ])
    image_bytes = _make_image(
        [("layer1.tar", layer1), ("layer2.tar", layer2), ("layer3.tar", layer3)]
    )

    monkeypatch.setattr(
        scanner, "export_image", lambda image, out_tar: out_tar.write_bytes(image_bytes)
    )
    result = scanner.scan("fake:whiteout-dangling")
    assert [v["member"] for v in result["violations"]] == ["etc/ssl/certs/pointer.pem"]


def test_scan_rejects_hardlink_to_genuine_ca_content_outside_trust_store(monkeypatch):
    """Defense in depth: even when the hardlink's ultimate target is
    genuinely verifiable CA content, the resolved member's own path must
    also be a trust-store location — a hardlink cannot borrow trust from
    real CA bytes stashed at an arbitrary app path. (The target itself,
    a `.pem` outside the allowlist, is independently flagged too — see
    test_real_ca_content_outside_trust_store_path_is_still_credential —
    so both members are expected violations here.)"""
    layer = _make_layer_bytes([
        ("app/vendor/real_ca_copy.pem", "reg", _REAL_CA_CERT_PEM),
        ("etc/ssl/certs/borrowed.pem", "hard", "app/vendor/real_ca_copy.pem"),
    ])
    image_bytes = _make_image([("layer.tar", layer)])

    monkeypatch.setattr(
        scanner, "export_image", lambda image, out_tar: out_tar.write_bytes(image_bytes)
    )
    result = scanner.scan("fake:untrusted-target-path")
    assert {v["member"] for v in result["violations"]} == {
        "etc/ssl/certs/borrowed.pem",
        "app/vendor/real_ca_copy.pem",
    }
    assert result["forbidden_count"] == 2


def test_scan_rejects_symlink_absolute_traversal_still_normalizes_outside_root(monkeypatch):
    """A trust-store symlink whose absolute target normalizes to a path
    escaping the image root (defense against a crafted `..`-laden absolute
    target) must not resolve."""
    layer = _make_layer_bytes([
        ("etc/ssl/certs/escape.pem", "sym", "/../../../../etc/shadow"),
    ])
    image_bytes = _make_image([("layer.tar", layer)])

    monkeypatch.setattr(
        scanner, "export_image", lambda image, out_tar: out_tar.write_bytes(image_bytes)
    )
    result = scanner.scan("fake:escape")
    assert [v["member"] for v in result["violations"]] == ["etc/ssl/certs/escape.pem"]
