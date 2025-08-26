apt -y install aria2

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin -d ./ -o ip-adapter-faceid-plusv2_sdxl.bin

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/1k3d68.onnx -d ./models/antelopev2 -o 1k3d68.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/2d106det.onnx -d ./models/antelopev2 -o 2d106det.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/genderage.onnx -d ./models/antelopev2 -o genderage.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/glintr100.onnx -d ./models/antelopev2 -o glintr100.onnx
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx -d ./models/antelopev2 -o scrfd_10g_bnkps.onnx

pip install -r requirments.txt

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin -d ./ -o ip-adapter_sd15.bin
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors -d ./models/image_encoder -o model.safetensors

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json -d ./models/image_encoder -o config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin -d ./models/image_encoder -o pytorch_model.bin
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin -d ./ -o ip-adapter-faceid_sd15.bin
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/InstantX/InstantID/raw/main/ControlNetModel/config.json -d ./checkpoints/ControlNetModel -o config.json

pip install git+https://github.com/tencent-ailab/IP-Adapter.git
