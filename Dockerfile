FROM node:22-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend ./
RUN npm run build


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    KYMO_EXPORT_DIR=/app/export \
    FRONTEND_DIST_DIR=/app/frontend/dist \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY alembic.ini .
COPY app ./app
COPY configs ./configs
COPY migrations ./migrations
COPY kymobutler_scripts ./kymobutler_scripts
COPY export ./export
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN test -f ./export/uni_seg.onnx \
    && test -f ./export/bi_seg.onnx \
    && test -f ./export/classifier.onnx \
    && test -f ./export/decision.onnx

RUN useradd --create-home --shell /usr/sbin/nologin waveatlas \
    && chown -R waveatlas:waveatlas /app

USER waveatlas

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
