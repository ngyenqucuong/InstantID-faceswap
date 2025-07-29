# InstantID Face Swap REST API

A REST API for performing high-quality face swapping using InstantID technology. This API allows you to swap faces between images while preserving pose, expression, and lighting.

## Features

- **High-quality face swapping** using InstantID and Stable Diffusion XL
- **RESTful API** with JSON and file upload support
- **Memory optimized** for GPU efficiency
- **Base64 and file upload** endpoints
- **Configurable parameters** for fine-tuning results
- **Health monitoring** and status endpoints
- **Docker support** for easy deployment

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- At least 8GB GPU memory

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd InstantID-faceswap
```

2. Install dependencies:
```bash
pip install -r requirements-api.txt
```

3. Download required models:
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download InstantID models
# Place your ip-adapter.bin and ControlNetModel in ./checkpoints/
```

4. Start the API server:
```bash
python api.py
```

The API will be available at `http://localhost:5000`

### Docker Installation

1. Build the Docker image:
```bash
docker build -t instantid-api .
```

2. Run the container:
```bash
docker run -p 5000:5000 --gpus all instantid-api
```

## API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2025-07-29T10:30:00"
}
```

### Face Swap (Base64)
```http
POST /face-swap
Content-Type: application/json
```

Request body:
```json
{
  "target_image": "base64_encoded_target_image",
  "reference_image": "base64_encoded_reference_image",
  "prompt": "high quality portrait",
  "steps": 4,
  "mask_strength": 0.8,
  "ip_adapter_scale": 0.7,
  "controlnet_conditioning_scale": 0.8,
  "guidance_scale": 1.0,
  "num_images": 2
}
```

Response:
```json
{
  "success": true,
  "result_images": ["base64_encoded_result_1", "base64_encoded_result_2"],
  "num_images": 2,
  "parameters_used": {
    "steps": 4,
    "mask_strength": 0.8,
    "ip_adapter_scale": 0.7,
    "num_images": 2
  },
  "timestamp": "2025-07-29T10:30:00"
}
```

### Face Swap (File Upload)
```http
POST /face-swap-file
Content-Type: multipart/form-data
```

Form data:
- `target_image`: Image file (PNG/JPG)
- `reference_image`: Image file (PNG/JPG)
- `prompt`: Text prompt (optional)
- `steps`: Number of inference steps (optional, default: 4)
- `mask_strength`: Mask strength (optional, default: 0.8)
- `ip_adapter_scale`: IP adapter scale (optional, default: 0.7)
- `num_images`: Number of result images (optional, default: 2)

Response: Returns a ZIP file containing all result images

### Models Status
```http
GET /models/status
```

Response:
```json
{
  "pipe_loaded": true,
  "face_app_loaded": true,
  "cuda_available": true,
  "cuda_device_count": 1
}
```

## Usage Examples

### Python Client

```python
from api_client import InstantIDAPI

# Initialize client
api = InstantIDAPI("http://localhost:5000")

# Perform face swap - now returns multiple images
result = api.face_swap_base64(
    target_image_path="target.jpg",
    reference_image_path="reference.jpg",
    steps=6,
    mask_strength=0.85,
    num_images=2
)

if result['success']:
    # Save all generated images
    for i, image in enumerate(result['images']):
        image.save(f"output_{i+1}.jpg")
    print(f"Generated {result['num_images']} images!")
else:
    print("Error:", result['error'])
```

### cURL Examples

#### Health check:
```bash
curl http://localhost:5000/health
```

#### Face swap with files (returns ZIP with multiple images):
```bash
curl -X POST http://localhost:5000/face-swap-file \
  -F "target_image=@target.jpg" \
  -F "reference_image=@reference.jpg" \
  -F "steps=4" \
  -F "mask_strength=0.8" \
  -F "num_images=2" \
  -o results.zip
```

### JavaScript/Fetch

```javascript
// File upload example - now returns ZIP with multiple images
const formData = new FormData();
formData.append('target_image', targetFile);
formData.append('reference_image', referenceFile);
formData.append('steps', '4');
formData.append('num_images', '2');

fetch('http://localhost:5000/face-swap-file', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(zipBlob => {
    // Handle ZIP file containing multiple result images
    const url = URL.createObjectURL(zipBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'face_swap_results.zip';
    a.click();
});
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "" | Text prompt for generation |
| `negative_prompt` | string | Auto | Negative prompt |
| `steps` | integer | 4 | Number of inference steps |
| `mask_strength` | float | 0.8 | Strength of face replacement (0.0-1.0) |
| `ip_adapter_scale` | float | 0.7 | IP adapter influence (0.0-1.0) |
| `controlnet_conditioning_scale` | float | 0.8 | ControlNet influence (0.0-1.0) |
| `guidance_scale` | float | 1.0 | Guidance scale for generation |
| `num_images` | integer | 2 | Number of result images to generate (1-4) |

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input)
- `500`: Internal server error

Error response format:
```json
{
  "error": "Error description",
  "timestamp": "2025-07-29T10:30:00"
}
```

## Performance Tips

1. **GPU Memory**: Ensure sufficient GPU memory (8GB+ recommended)
2. **Image Size**: Smaller images process faster (1024x1024 max recommended)
3. **Batch Processing**: Process images sequentially to avoid memory issues
4. **Model Caching**: Models are loaded once on startup for efficiency

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce image resolution
   - Lower inference steps
   - Restart the API to clear memory

2. **No Face Detected**:
   - Ensure faces are clearly visible
   - Check image quality and lighting
   - Face should be reasonably large in the image

3. **Model Loading Errors**:
   - Verify checkpoints are in the correct directory
   - Check file permissions
   - Ensure all dependencies are installed

### Logs

The API logs important information to the console. Monitor these logs for debugging:

```bash
python api.py
# or
docker logs <container_id>
```

## Security Considerations

- **Input Validation**: All images are validated before processing
- **File Size Limits**: Implement reasonable file size limits in production
- **Rate Limiting**: Consider adding rate limiting for production use
- **HTTPS**: Use HTTPS in production environments

## License

This project uses the same license as the original InstantID implementation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Create an issue with detailed information
