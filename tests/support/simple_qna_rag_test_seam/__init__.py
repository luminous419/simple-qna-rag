"""Test-only seam package — intentionally outside `src/`.

The production Dockerfile's `production` stage COPYs only `src/`,
`pyproject.toml`, `README.md`, `LICENSE`, `web/static/`, `web/templates/`
(Design.md §7.1). `tests/` is never COPYed into that stage, so this package
is physically absent from the production image (Design.md §5.2-a,
DR-I3-MAJ-02).
"""
