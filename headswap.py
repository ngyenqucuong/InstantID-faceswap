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

# Add MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image

# Initialize MediaPipe
mp_image = mp.Image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# MediaPipe segmentation model
def initialize_mediapipe():
    """Initialize MediaPipe selfie segmentation"""
    return mp_selfie_segmentation.SelfieSegmentation(model_selection=1)  # model_selection=1 for better quality

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


def get_hair_and_face_masks_mediapipe(image, face_bbox):
    """
    Use MediaPipe to create precise hair and face masks
    
    Args:
        image: PIL Image
        face_bbox: Face bounding box from InsightFace [x1, y1, x2, y2]
    
    Returns:
        hair_mask: PIL Image of hair mask
        face_mask: PIL Image of face mask  
        combined_mask: PIL Image of combined head mask
    """
    # Initialize MediaPipe
    segmentation = initialize_mediapipe()
    
    # Convert PIL to cv2
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Process the image
    results = segmentation.process(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
    
    # Get segmentation mask
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.3
    full_person_mask = condition[:, :, 0].astype(np.uint8) * 255
    
    # Create face mask from bounding box
    x1, y1, x2, y2 = face_bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    height, width = full_person_mask.shape
    face_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Expand face region slightly for better coverage
    face_expand = 0.1  # 10% expansion
    face_width = x2 - x1
    face_height = y2 - y1
    
    expanded_x1 = max(0, int(x1 - face_width * face_expand))
    expanded_y1 = max(0, int(y1 - face_height * face_expand))
    expanded_x2 = min(width, int(x2 + face_width * face_expand))
    expanded_y2 = min(height, int(y2 + face_height * face_expand))
    
    face_mask[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = 255
    
    # Create hair mask by subtracting face from person mask
    hair_mask = full_person_mask.copy()
    
    # Remove face region from hair mask
    face_region_expanded = face_mask > 0
    hair_mask[face_region_expanded] = 0
    
    # Only keep upper portion for hair (remove body parts)
    hair_cutoff_y = int(y2 + face_height * 0.5)  # Cut below face
    hair_mask[hair_cutoff_y:, :] = 0
    
    # Refine hair mask - keep only upper region of person segmentation
    face_center_y = (y1 + y2) // 2
    hair_top_region = slice(0, max(face_center_y, hair_cutoff_y))
    hair_mask_refined = np.zeros_like(hair_mask)
    hair_mask_refined[hair_top_region, :] = hair_mask[hair_top_region, :]
    
    # Combine face and hair masks
    combined_mask = np.maximum(face_mask, hair_mask_refined)
    
    # Apply morphological operations for smoother masks
    kernel = np.ones((5, 5), np.uint8)
    
    # Smooth face mask
    face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)
    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
    
    # Smooth hair mask
    hair_mask_refined = cv2.morphologyEx(hair_mask_refined, cv2.MORPH_CLOSE, kernel)
    hair_mask_refined = cv2.GaussianBlur(hair_mask_refined, (11, 11), 0)
    
    # Smooth combined mask
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.GaussianBlur(combined_mask, (13, 13), 0)
    
    # Convert to PIL Images
    hair_mask_pil = Image.fromarray(hair_mask_refined)
    face_mask_pil = Image.fromarray(face_mask)
    combined_mask_pil = Image.fromarray(combined_mask)
    
    # Cleanup MediaPipe
    segmentation.close()
    
    return hair_mask_pil, face_mask_pil, combined_mask_pil

def prepareMediaPipeHeadMaskAndPose(pose_image, face_info, padding=80, resize=True, use_combined_mask=True):
    """
    Prepare mask and control images using MediaPipe segmentation
    
    Args:
        pose_image: PIL Image of target
        face_info: Face detection info from InsightFace
        padding: Padding around the region
        resize: Whether to resize the output
        use_combined_mask: Use combined head mask or just hair mask
    """
    kps = face_info['kps'].copy()
    width, height = pose_image.size
    
    x1, y1, x2, y2 = face_info['bbox']
    
    # Get MediaPipe masks
    hair_mask, face_mask, combined_mask = get_hair_and_face_masks_mediapipe(pose_image, [x1, y1, x2, y2])
    
    # Choose which mask to use
    if use_combined_mask:
        mask_to_use = combined_mask
        print("Using combined hair + face mask")
    else:
        mask_to_use = hair_mask
        print("Using hair-only mask")
    
    # Find mask boundaries for cropping
    mask_array = np.array(mask_to_use)
    mask_coords = np.where(mask_array > 0)
    
    if len(mask_coords[0]) == 0:
        # Fallback to face bbox if no mask found
        print("Warning: No mask detected, falling back to face region")
        m_y1, m_x1 = int(y1), int(x1)
        m_y2, m_x2 = int(y2), int(x2)
    else:
        m_y1, m_x1 = mask_coords[0].min(), mask_coords[1].min()
        m_y2, m_x2 = mask_coords[0].max(), mask_coords[1].max()
    
    # Add padding
    p_x1 = max(0, m_x1 - padding)
    p_y1 = max(0, m_y1 - padding)
    p_x2 = min(width, m_x2 + padding)
    p_y2 = min(height, m_y2 + padding)
    
    # Crop mask and image
    mask_crop = mask_array[p_y1:p_y2, p_x1:p_x2]
    mask_pil = Image.fromarray(mask_crop)
    
    # Apply additional smoothing
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
    
    # Save intermediate masks for debugging
    hair_mask.save('debug_hair_mask.jpg')
    face_mask.save('debug_face_mask.jpg')
    combined_mask.save('debug_combined_mask.jpg')
    
    return (mask_pil, image, control_image), (p_x1, p_y1, original_width, original_height)

# // ...existing code for resize_img and prepare_enhanced_face_embedding...
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
# pipe.enable_attention_slicing()
# # pipe.enable_sequential_cpu_offload()  # This moves models to CPU when not in use
# pipe.enable_vae_slicing()
# pipe = pipe.to("cuda")

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

# Use MediaPipe for precise head mask
print("Creating precise head mask with MediaPipe...")
images, position = prepareMediaPipeHeadMaskAndPose(
    pose_image,
    face_info,
    80,    # padding
    True,  # resize
    True   # use_combined_mask (True for hair+face, False for hair only)
)
mask, pose_image_preprocessed, control_image = images

face_emb = prepare_enhanced_face_embedding([
    'examples/sess_9m7a6q7b7.jpg',
    # Add more reference images of the same person for better results:
    # 'path/to/another/reference/image.jpg',
])


# Better prompts for head swapping
prompt = 'high quality, detailed, sharp, photorealistic portrait'
negative_prompt = '''(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting,
drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured,
blurry, artifacts, bad anatomy, bad proportions, extra limbs, cloned face,
malformed limbs, gross proportions, missing arms, missing legs, extra arms,
extra legs, fused fingers, too many fingers, long neck, bad hair, weird hair,
floating hair, disconnected hair, hair artifacts'''

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

print("MediaPipe-enhanced head swap completed!")
print("Check debug masks: debug_hair_mask.jpg, debug_face_mask.jpg, debug_combined_mask.jpg")
print("Final result: result_head_swap.jpg")