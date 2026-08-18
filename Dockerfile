FROM python:3.14-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen --no-install-project

COPY olist_code/ ./olist_code/
COPY docker_entrypoint.py ./
RUN uv sync --no-dev --frozen

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PORT=3080
EXPOSE 3080

CMD [".venv/bin/python", "docker_entrypoint.py"]
