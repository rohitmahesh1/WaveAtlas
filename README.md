# WaveAtlas

WaveAtlas is a local-first data analysis and visualization platform for researchers.
The current app supports tabular uploads, runs a wave/track extraction pipeline in a FastAPI backend, and streams processed tracks to a React viewer for inspection.

## Active Structure

- `app/`: FastAPI backend, pipeline orchestration, signal processing, artifact storage, and job persistence.
- `frontend/`: React + Vite viewer for uploads, live run monitoring, filtering, and track inspection.
- `configs/`: default pipeline configuration.
- `docs/`: user-facing configuration reference.
- `samples/`: small example inputs that are safe to keep in Git.
- `scripts/`: helper scripts for local development and fetching large release assets.
- `kymobutler_scripts/`: Wolfram scripts used by the legacy KymoButler integration path.

## Large Assets

The ONNX model binaries are intentionally not committed in the main Git history.
For now, treat them as release assets or future Git LFS candidates.

To fetch model files into `export/`:

```bash
bash ./scripts/fetch-models.sh
```

To fetch sample assets:

```bash
bash ./scripts/fetch-samples.sh
```

## Local Development

Start the backend:

```bash
bash ./scripts/dev-backend.sh
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```
