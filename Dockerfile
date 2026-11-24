# TensorFold XPU serving image for Intel Arc Pro B70 — engine + grammar lib only, no weights.
FROM intel/llm-scaler-vllm@sha256:a397e04d66494637862b1927ee92ab4dd8b2f03f0dc9d359ec9536959a88e704

COPY tensorfold-xpu /opt/tensorfold
COPY README.md /opt/tensorfold/README.md

ENV PATH=/opt/tensorfold/bin:$PATH \
    ZE_AFFINITY_MASK=0 \
    XPU_KV_INT8=1

EXPOSE 8000
WORKDIR /opt/tensorfold
ENTRYPOINT ["/opt/tensorfold/bin/tensorfold-native"]
CMD ["serve", "/models/model", "--host", "0.0.0.0", "--port", "8000", "--context", "200000", "--prompt-cache-gib", "0.25", "--no-thinking"]
