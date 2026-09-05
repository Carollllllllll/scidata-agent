# Build the React workbench first, then copy only its static output into the
# runtime image. The final image contains no Node.js or frontend source.
FROM node:22-bookworm-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    SCIDATA_RUNTIME_DIR=/data/runtime \
    SCIDATA_STATIC_DIR=/app/frontend/dist
WORKDIR /app/backend

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/scidata_agent ./scidata_agent
COPY --from=frontend-build /build/frontend/dist /app/frontend/dist

RUN mkdir -p /data/runtime
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn scidata_agent.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
