apt -y install -qq aria2
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantID/raw/main/ControlNetModel/config.json -d ./checkpoints/ControlNetModel -o config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors -d ./checkpoints/ControlNetModel -o diffusion_pytorch_model.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin -d ./checkpoints -o ip-adapter.bin

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/1k3d68.onnx -d ./models/antelopev2 -o 1k3d68.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/2d106det.onnx -d ./models/antelopev2 -o 2d106det.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/genderage.onnx -d ./models/antelopev2 -o genderage.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/glintr100.onnx -d ./models/antelopev2 -o glintr100.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx -d ./models/antelopev2 -o scrfd_10g_bnkps.onnx

pip install -r requirments.txt