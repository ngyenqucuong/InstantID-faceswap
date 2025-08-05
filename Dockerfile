# Use official Python runtime as base image
FROM cnstark/pytorch:2.2.0-py3.10.15-cuda12.1.0-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    aria2 

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantID/raw/main/ControlNetModel/config.json -d ./checkpoints/ControlNetModel -o config.json\
&& aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors -d ./checkpoints/ControlNetModel -o diffusion_pytorch_model.safetensors\
&& aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin -d ./checkpoints -o ip-adapter.bin\
&& aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/1k3d68.onnx -d ./models/antelopev2 -o 1k3d68.onnx\
&& aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/2d106det.onnx -d ./models/antelopev2 -o 2d106det.onnx\
&& aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/genderage.onnx -d ./models/antelopev2 -o genderage.onnx\
&& aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/glintr100.onnx -d ./models/antelopev2 -o glintr100.onnx\
&& aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx -d ./models/antelopev2 -o scrfd_10g_bnkps.onnx
# Copy requirements first for better caching
COPY requirements.txt . 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results checkpoints

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Run the application
CMD ["python", "api.py"]
