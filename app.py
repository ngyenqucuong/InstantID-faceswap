import gc
import torch
import os

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Enable memory efficient attention
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler
import math
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import draw_kps
from pipeline_stable_diffusion_xl_instantid_inpaint import StableDiffusionXLInstantIDInpaintPipeline
from huggingface_hub import hf_hub_download

from PIL import Image

def resize_img(input_image, max_side=1024, min_side=768, size=None,  # Back to higher resolution
               pad_to_max_side=False, mode=Image.LANCZOS, base_pixel_number=64):  # Better resampling
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

# Enhanced face embedding preparation with multiple reference images
def prepare_enhanced_face_embedding(face_list):
    face_embeddings = []
    for face_path in face_list:
        face_image = load_image(face_path)

        # Enhance image quality before processing
        face_image = ImageEnhance.Sharpness(face_image).enhance(1.2)
        face_image = ImageEnhance.Contrast(face_image).enhance(1.1)

        face_image = resize_img(face_image)
        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))

        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
            face_emb = face_info['embedding']
            face_embeddings.append(face_emb)

        del face_image, face_info
        gc.collect()

    if len(face_embeddings) == 0:
        raise ValueError("No faces detected in reference images")

    # Average multiple embeddings for better consistency
    return np.mean(face_embeddings, axis=0)

# Improved mask and pose preparation
def prepareMaskAndPoseAndControlImage(pose_image, face_info, padding=80, mask_grow=30, resize=True):
    if padding < mask_grow:
        raise ValueError('mask_grow cannot be greater than padding')

    kps = face_info['kps'].copy()
    width, height = pose_image.size

    x1, y1, x2, y2 = face_info['bbox']
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Enhanced mask creation with feathering
    m_x1 = max(0, x1 - mask_grow)
    m_y1 = max(0, y1 - mask_grow)
    m_x2 = min(width, x2 + mask_grow)
    m_y2 = min(height, y2 + mask_grow)

    p_x1 = max(0, x1 - padding)
    p_y1 = max(0, y1 - padding)
    p_x2 = min(width, x2 + padding)
    p_y2 = min(height, y2 + padding)

    # Create smooth mask with gradient edges
    mask = np.zeros([height, width], dtype=np.float32)

    # Inner solid region
    inner_grow = mask_grow // 2
    inner_x1 = max(0, x1 - inner_grow)
    inner_y1 = max(0, y1 - inner_grow)
    inner_x2 = min(width, x2 + inner_grow)
    inner_y2 = min(height, y2 + inner_grow)

    mask[inner_y1:inner_y2, inner_x1:inner_x2] = 1.0

    # Outer transition region
    mask[m_y1:m_y2, m_x1:m_x2] = 0.7

    # Convert to PIL and apply Gaussian blur for smooth edges
    mask_crop = mask[p_y1:p_y2, p_x1:p_x2]
    mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=2))

    # Crop the pose image
    image = np.array(pose_image)[p_y1:p_y2, p_x1:p_x2]
    image = Image.fromarray(image.astype(np.uint8))

    # Adjust keypoints
    original_width, original_height = image.size
    kps -= [p_x1, p_y1]

    if resize:
        mask_pil = resize_img(mask_pil)
        image = resize_img(image)
        new_width, new_height = image.size
        kps *= [new_width / original_width, new_height / original_height]

    # Create control image with enhanced keypoints
    control_image = draw_kps(image, kps)

    return (mask_pil, image, control_image), (p_x1, p_y1, original_width, original_height)

app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Clear memory before loading models
torch.cuda.empty_cache()
gc.collect()

# Load models with memory optimizations
face_adapter = f'./checkpoints/ip-adapter.bin'
controlnet_path = f'./checkpoints/ControlNetModel'

# Load pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_lora.safetensors"

pipe = StableDiffusionXLInstantIDInpaintPipeline.from_pretrained(
    base,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

# Enable memory efficient attention and CPU offloading
pipe.enable_attention_slicing()
# pipe.enable_sequential_cpu_offload()  # This moves models to CPU when not in use
pipe.enable_vae_slicing()
pipe = pipe.to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Load adapters
pipe.load_ip_adapter_instantid(face_adapter)
pipe.load_lora_weights(hf_hub_download(repo, ckpt))
pipe.fuse_lora()

# Clear memory after model loading
torch.cuda.empty_cache()
gc.collect()


# Load and prepare your target image
pose_image = load_image('examples/poses/target.png')
# Enhance the input image quality
pose_image = ImageEnhance.Sharpness(pose_image).enhance(1.1)
pose_image = ImageEnhance.Contrast(pose_image).enhance(1.05)

# Detect face with better parameters
face_info = app.get(cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR))

if len(face_info) == 0:
    raise ValueError("No face detected in target image")

face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]

# Prepare mask and control images with better parameters
images, position = prepareMaskAndPoseAndControlImage(
    pose_image,
    face_info,
    80,   # Increased padding for better context
    30,   # Better mask grow
    True
)
mask, pose_image_preprocessed, control_image = images

# Clear memory before inference
torch.cuda.empty_cache()
gc.collect()



# Enhanced face embeddings (use multiple reference images if available)
face_emb = prepare_enhanced_face_embedding([
    'examples/sess_9m7a6q7b7.jpg',

])

# Clear memory after face embedding
torch.cuda.empty_cache()
gc.collect()


# Better generation parameters
prompt = ''
negative_prompt = '''(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting,
drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured,
blurry, artifacts, bad anatomy, bad proportions, extra limbs, cloned face,
malformed limbs, gross proportions, missing arms, missing legs, extra arms,
extra legs, fused fingers, too many fingers, long neck'''

steps = 4  # Increased steps for better quality
mask_strength = 0.8  # Higher strength for better face replacement

# Ensure all inputs are on the correct device
device = "cuda"
if isinstance(face_emb, np.ndarray):
    face_emb = torch.from_numpy(face_emb).to(device, dtype=torch.float16)

print("Starting high-quality inference...")

# Generate with optimal parameters
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image_embeds=face_emb,
    control_image=control_image,
    image=pose_image_preprocessed,
    mask_image=mask,
    controlnet_conditioning_scale=0.8,  # Higher for better control
    strength=mask_strength,
    ip_adapter_scale=0.7,  # Increased for stronger face identity
    num_inference_steps=int(math.ceil(steps / mask_strength)),
    guidance_scale=1.0,  # Slight guidance for better quality
    # generator=torch.Generator(device=device).manual_seed(42)
).images[0]

# Save the result directly
image.save('face_processed.jpg')

# Apply final enhancement
final_result = ImageEnhance.Sharpness(image).enhance(1.05)
final_result.save('result_improved.jpg')

print("High-quality face swap completed!")
print("Check 'result_improved.jpg' for the final result")

# Clear memory
torch.cuda.empty_cache()
gc.collect()