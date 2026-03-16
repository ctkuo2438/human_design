# GLIBC = GNU C Library
# Base image — python:3.11-slim gives you a lightweight Python environment with a modern enough GLIBC version to support packages like FAISS.

FROM python:3.11-slim

RUN pip install --no-cache-dir \
awslambdaric \
"numpy<2" \
"faiss-cpu==1.9.0.post1" \
sentence-transformers \
boto3

# Pre-download model to a known path
ENV SENTENCE_TRANSFORMERS_HOME=/opt/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

WORKDIR /var/task
COPY lambda_function.py .

ENV SENTENCE_TRANSFORMERS_HOME=/opt/models

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambda_function.lambda_handler"]