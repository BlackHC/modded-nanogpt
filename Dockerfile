FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7
ENV PATH=/usr/local/bin:$PATH

# Create directories and copy data files early for better layer caching
RUN mkdir -p /modded-nanogpt/data/fineweb10B
COPY data/fineweb10B/*.bin /modded-nanogpt/data/fineweb10B/

RUN apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12 /usr/local/bin/pip

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock /modded-nanogpt/
WORKDIR /modded-nanogpt

# Install dependencies using uv
RUN uv sync --frozen

# # Copy the rest of the application code
# COPY . /modded-nanogpt/

CMD ["bash"]
ENTRYPOINT []
