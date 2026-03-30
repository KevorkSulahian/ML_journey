# .devcontainer/Dockerfile (or at repo root and refer to it)
FROM mambaorg/micromamba:1.5.8-jammy

# Create an env with a consistent BLAS stack (MKL or OpenBLAS). Here: MKL.
COPY --chown=mambauser:mambauser environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean -a -y

# Jupyter will run inside container; no token for local dev
ENV JUPYTER_TOKEN=""
EXPOSE 8888

# Default to bash; VS Code handles starting kernels
SHELL ["/bin/bash", "-lc"]
