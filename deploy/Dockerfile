# syntax=docker/dockerfile:1
# M4.3-REQ-005 — minimal-privilege Linux CPU OCI image (Design.md §7.1).
ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

FROM base AS builder
RUN python -m pip install --no-cache-dir "uv==0.8.15"
COPY requirements.lock ./
RUN python -m pip install --require-hashes --no-cache-dir \
      -r requirements.lock \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --target /install
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir --no-deps --target /install -e .

# test stage: proves install + import succeed; pytest itself runs in the
# hosted python-tests job (build/test responsibility split, REQ-005.5).
FROM base AS test
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY pyproject.toml README.md LICENSE requirements.lock ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY evaluation/ ./evaluation/
RUN python -c "from simple_qna_rag.web.server import app"
CMD ["python", "-c", "print('test stage import smoke only')"]

# production: non-root, no builder/compiler/test/evaluation/runtime data.
FROM base AS production
RUN groupadd -g 10001 app && \
    useradd -u 10001 -g app -M -s /usr/sbin/nologin app
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY web/static/ ./web/static/
COPY web/templates/ ./web/templates/
RUN mkdir -p /app/runtime/index && chown -R app:app /app/runtime
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "-m", "uvicorn", "simple_qna_rag.web.server:app", \
            "--host", "0.0.0.0", "--port", "8000"]
