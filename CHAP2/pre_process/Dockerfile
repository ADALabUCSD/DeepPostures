FROM python:3.11.10-slim

LABEL org.opencontainers.image.source="https://github.com/ADALabUCSD/DeepPostures"
LABEL org.opencontainers.image.description="CHAP2 SOL/PASOS CPU preprocessing runtime"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.13.1 \
    h5py==3.10.0

WORKDIR /opt/chap2-preprocess
COPY pre_process_data.py run_sol_pasos.py summarize_sol_pasos.py test_pre_process_data.py test_run_sol_pasos.py ./
RUN python -m unittest -v test_pre_process_data test_run_sol_pasos

ENTRYPOINT ["python", "-u", "/opt/chap2-preprocess/pre_process_data.py"]
