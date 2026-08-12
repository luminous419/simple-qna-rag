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
    reviewer reproduced against scan()'s actual call site."""
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
    categories = {v["category"] for v in result["violations"]}
    assert "credential" in categories
    assert result["forbidden_count"] >= 1
