import gc
import torch
import os
import base64
import io
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global variables for models (loaded once)
pipe = None
face_app = None

def resize_img(input_image, max_side=1024, min_side=768, size=None,
               pad_to_max_side=False, mode=Image.LANCZOS, base_pixel_number=64):
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

def prepare_enhanced_face_embedding(face_image):
    """Prepare face embedding from a single image"""
    face_image = ImageEnhance.Sharpness(face_image).enhance(1.2)
    face_image = ImageEnhance.Contrast(face_image).enhance(1.1)
    face_image = resize_img(face_image)
    
    face_info = face_app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    
    if len(face_info) == 0:
        raise ValueError("No face detected in reference image")
    
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    face_emb = face_info['embedding']
    
    del face_image, face_info
    gc.collect()
    
    return face_emb

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image.convert('RGB')

def initialize_models():
    """Initialize models once on startup"""
    global pipe, face_app
    
    print("Initializing models...")
    
    # Initialize face analysis
    face_app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

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
    
    print("Models initialized successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': pipe is not None and face_app is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/face-swap', methods=['POST'])
def face_swap():
    """Main face swap endpoint"""
    try:
        # Check if models are initialized
        if pipe is None or face_app is None:
            return jsonify({'error': 'Models not initialized'}), 500
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['target_image', 'reference_image']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Decode images from base64
        try:
            target_image = base64_to_image(data['target_image'])
            reference_image = base64_to_image(data['reference_image'])
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Get optional parameters
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '''(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting,
drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured,
blurry, artifacts, bad anatomy, bad proportions, extra limbs, cloned face,
malformed limbs, gross proportions, missing arms, missing legs, extra arms,
extra legs, fused fingers, too many fingers, long neck''')
        
        steps = data.get('steps', 4)
        mask_strength = data.get('mask_strength', 0.8)
        ip_adapter_scale = data.get('ip_adapter_scale', 0.7)
        controlnet_conditioning_scale = data.get('controlnet_conditioning_scale', 0.8)
        guidance_scale = data.get('guidance_scale', 1.0)
        num_images = data.get('num_images', 2)  # Generate 2 images by default
        
        # Enhance input images
        target_image = ImageEnhance.Sharpness(target_image).enhance(1.1)
        target_image = ImageEnhance.Contrast(target_image).enhance(1.05)
        
        # Detect face in target image
        face_info = face_app.get(cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR))
        
        if len(face_info) == 0:
            return jsonify({'error': 'No face detected in target image'}), 400
        
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        
        # Prepare mask and control images
        images, position = prepareMaskAndPoseAndControlImage(
            target_image,
            face_info,
            80,   # padding
            30,   # mask_grow
            True
        )
        mask, pose_image_preprocessed, control_image = images
        
        # Clear memory before inference
        torch.cuda.empty_cache()
        gc.collect()
        
        # Prepare face embedding
        face_emb = prepare_enhanced_face_embedding(reference_image)
        
        # Clear memory after face embedding
        torch.cuda.empty_cache()
        gc.collect()
        
        # Ensure all inputs are on the correct device
        device = "cuda"
        if isinstance(face_emb, np.ndarray):
            face_emb = torch.from_numpy(face_emb).to(device, dtype=torch.float16)
        
        print("Starting face swap inference...")
        
        # Generate with optimal parameters
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            control_image=control_image,
            image=pose_image_preprocessed,
            mask_image=mask,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=mask_strength,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=int(math.ceil(steps / mask_strength)),
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
        ).images
        
        # Apply final enhancement to all images
        enhanced_images = []
        result_base64_list = []
        
        for i, image in enumerate(images):
            final_result = ImageEnhance.Sharpness(image).enhance(1.05)
            enhanced_images.append(final_result)
            
            # Convert result to base64
            result_base64 = image_to_base64(final_result)
            result_base64_list.append(result_base64)
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return jsonify({
            'success': True,
            'result_images': result_base64_list,  # Return list of images
            'num_images': len(result_base64_list),
            'parameters_used': {
                'steps': steps,
                'mask_strength': mask_strength,
                'ip_adapter_scale': ip_adapter_scale,
                'controlnet_conditioning_scale': controlnet_conditioning_scale,
                'guidance_scale': guidance_scale,
                'num_images': num_images
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in face_swap: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Face swap failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/face-swap-file', methods=['POST'])
def face_swap_file():
    """Face swap endpoint that accepts file uploads"""
    try:
        # Check if models are initialized
        if pipe is None or face_app is None:
            return jsonify({'error': 'Models not initialized'}), 500
        
        # Check if files are present
        if 'target_image' not in request.files or 'reference_image' not in request.files:
            return jsonify({'error': 'Missing target_image or reference_image file'}), 400
        
        target_file = request.files['target_image']
        reference_file = request.files['reference_image']
        
        # Check if files are valid
        if target_file.filename == '' or reference_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not (allowed_file(target_file.filename) and allowed_file(reference_file.filename)):
            return jsonify({'error': 'Invalid file format. Allowed: png, jpg, jpeg'}), 400
        
        # Load images
        target_image = Image.open(target_file.stream).convert('RGB')
        reference_image = Image.open(reference_file.stream).convert('RGB')
        
        # Get optional parameters from form data
        prompt = request.form.get('prompt', '')
        steps = int(request.form.get('steps', 4))
        mask_strength = float(request.form.get('mask_strength', 0.8))
        ip_adapter_scale = float(request.form.get('ip_adapter_scale', 0.7))
        num_images = int(request.form.get('num_images', 2))  # Generate 2 images by default
        
        # Process face swap (same logic as above)
        target_image = ImageEnhance.Sharpness(target_image).enhance(1.1)
        target_image = ImageEnhance.Contrast(target_image).enhance(1.05)
        
        face_info = face_app.get(cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR))
        
        if len(face_info) == 0:
            return jsonify({'error': 'No face detected in target image'}), 400
        
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        
        images, position = prepareMaskAndPoseAndControlImage(target_image, face_info, 80, 30, True)
        mask, pose_image_preprocessed, control_image = images
        
        torch.cuda.empty_cache()
        gc.collect()
        
        face_emb = prepare_enhanced_face_embedding(reference_image)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        device = "cuda"
        if isinstance(face_emb, np.ndarray):
            face_emb = torch.from_numpy(face_emb).to(device, dtype=torch.float16)
        
        images = pipe(
            prompt=prompt,
            negative_prompt='''(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting,
drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured,
blurry, artifacts, bad anatomy, bad proportions, extra limbs, cloned face,
malformed limbs, gross proportions, missing arms, missing legs, extra arms,
extra legs, fused fingers, too many fingers, long neck''',
            image_embeds=face_emb,
            control_image=control_image,
            image=pose_image_preprocessed,
            mask_image=mask,
            controlnet_conditioning_scale=0.8,
            strength=mask_strength,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=int(math.ceil(steps / mask_strength)),
            guidance_scale=1.0,
            num_images_per_prompt=num_images,
        ).images
        
        # Create a ZIP file containing all results
        import zipfile
        import tempfile
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f"face_swap_results_{uuid.uuid4().hex}.zip")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for i, image in enumerate(images):
                final_result = ImageEnhance.Sharpness(image).enhance(1.05)
                
                # Save each image to temp file
                result_filename = f"result_{i+1}_{uuid.uuid4().hex}.jpg"
                result_path = os.path.join(temp_dir, result_filename)
                final_result.save(result_path)
                
                # Add to ZIP
                zipf.write(result_path, result_filename)
                
                # Clean up individual file
                os.remove(result_path)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return send_file(zip_path, as_attachment=True, download_name=f"face_swap_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        
    except Exception as e:
        print(f"Error in face_swap_file: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Face swap failed: {str(e)}'}), 500

@app.route('/models/status', methods=['GET'])
def models_status():
    """Get model loading status"""
    return jsonify({
        'pipe_loaded': pipe is not None,
        'face_app_loaded': face_app is not None,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })

if __name__ == '__main__':
    print("Starting InstantID Face Swap API...")
    
    # Initialize models on startup
    initialize_models()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
