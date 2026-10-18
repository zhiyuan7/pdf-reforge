FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MINERU_HOME=/cache/mineru \
    HF_HOME=/cache/huggingface \
    MODELSCOPE_CACHE=/cache/modelscope \
    XDG_CACHE_HOME=/cache/xdg \
    MINERU_DEVICE_MODE=cpu \
    MINERU_MODEL_SMALL_BACKEND=onnx \
    MINERU_MODEL_VLM_ENGINE=llama-cpp

# OpenCV, the OCR runtime, XeTeX and Chinese fonts need these system libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates perl fontconfig fonts-noto-cjk \
        libgl1 libglib2.0-0 libgomp1 libfontconfig1 libfreetype6 libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-web.txt requirements-mineru.txt ./
RUN python -m venv /opt/venv \
    && python -m pip install --no-cache-dir -r requirements-web.txt -r requirements-mineru.txt \
    && ln -s /opt/venv /app/.venv

# Keep the expensive dependency layers separate from application changes.
COPY install_book_tex.py ./
# tlmgr delegates HTTPS downloads to curl, which Python's slim image omits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
ARG TINYTEX_RELEASE=v2026.10
ARG TEX_REPOSITORY=https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/tlnet
RUN python install_book_tex.py --release "${TINYTEX_RELEASE}" --repository "${TEX_REPOSITORY}"

# NVIDIA's Vulkan driver needs EGL even for headless llama.cpp inference.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY main.py serve.py run_mineru.py mineru_checkpoint.py pipeline_progress.py ./
COPY book_pipeline/ ./book_pipeline/
COPY book_templates/ ./book_templates/
COPY docker/ ./docker/
COPY models/font_candidates/LxgwWenkaiGB/ ./models/font_candidates/LxgwWenkaiGB/
COPY book_config.json book_metadata.example.yaml ./

RUN mkdir -p /app/input_pdfs /app/output /cache \
    && python main.py --check-local \
    && python -c "from mineru.parser.tier import ensure_tier_runtime_dependencies; ensure_tier_runtime_dependencies('standard')"

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:8765/api/environment', timeout=3).read()"]

CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8765"]
