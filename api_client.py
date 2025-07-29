import requests
import base64
import json
from datetime import datetime
from PIL import Image
import io

class InstantIDAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def image_to_base64(self, image_path):
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def base64_to_image(self, base64_string):
        """Convert base64 string to PIL Image"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    def health_check(self):
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def face_swap_base64(self, target_image_path, reference_image_path, **kwargs):
        """
        Perform face swap using base64 encoded images
        
        Args:
            target_image_path: Path to target image (pose source)
            reference_image_path: Path to reference image (face source)
            **kwargs: Optional parameters (prompt, steps, mask_strength, etc.)
        """
        # Convert images to base64
        target_b64 = self.image_to_base64(target_image_path)
        reference_b64 = self.image_to_base64(reference_image_path)
        
        # Prepare request data
        data = {
            "target_image": target_b64,
            "reference_image": reference_b64,
            **kwargs  # Include any additional parameters
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/face-swap",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    # Convert all result images back to PIL Images
                    result_images = []
                    for img_b64 in result['result_images']:
                        result_image = self.base64_to_image(img_b64)
                        result_images.append(result_image)
                    
                    return {
                        'success': True,
                        'images': result_images,  # List of PIL Images
                        'num_images': result.get('num_images', len(result_images)),
                        'parameters': result.get('parameters_used', {}),
                        'timestamp': result.get('timestamp')
                    }
                else:
                    return {'success': False, 'error': result.get('error')}
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def face_swap_files(self, target_image_path, reference_image_path, output_dir=None, **kwargs):
        """
        Perform face swap using file uploads
        
        Args:
            target_image_path: Path to target image
            reference_image_path: Path to reference image
            output_dir: Optional directory to save results (if None, returns images)
            **kwargs: Optional parameters
        """
        try:
            # Prepare files
            files = {
                'target_image': open(target_image_path, 'rb'),
                'reference_image': open(reference_image_path, 'rb')
            }
            
            # Prepare form data
            data = {}
            for key, value in kwargs.items():
                data[key] = str(value)
            
            response = requests.post(
                f"{self.base_url}/face-swap-file",
                files=files,
                data=data
            )
            
            # Close files
            files['target_image'].close()
            files['reference_image'].close()
            
            if response.status_code == 200:
                # The response is now a ZIP file containing multiple images
                if output_dir:
                    import zipfile
                    import tempfile
                    import os
                    
                    # Save ZIP file
                    zip_path = os.path.join(output_dir, f"face_swap_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Extract images
                    extracted_files = []
                    with zipfile.ZipFile(zip_path, 'r') as zipf:
                        zipf.extractall(output_dir)
                        extracted_files = zipf.namelist()
                    
                    return {
                        'success': True, 
                        'zip_path': zip_path,
                        'extracted_files': [os.path.join(output_dir, f) for f in extracted_files],
                        'num_images': len(extracted_files)
                    }
                else:
                    # Return ZIP content as bytes
                    return {'success': True, 'zip_content': response.content}
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                return {'success': False, 'error': error_data}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_models_status(self):
        """Get model loading status"""
        try:
            response = requests.get(f"{self.base_url}/models/status")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Initialize API client
    api = InstantIDAPI("http://localhost:5000")
    
    # Check health
    print("Health check:", api.health_check())
    
    # Check models status
    print("Models status:", api.get_models_status())
    
    # Example face swap with base64 (more flexible)
    result = api.face_swap_base64(
        target_image_path="examples/poses/target.png",
        reference_image_path="examples/sess_9m7a6q7b7.jpg",
        prompt="high quality portrait",
        steps=6,
        mask_strength=0.85,
        ip_adapter_scale=0.8,
        num_images=2
    )
    
    if result['success']:
        # Save all results
        for i, image in enumerate(result['images']):
            image.save(f"api_result_base64_{i+1}.jpg")
        print(f"Base64 face swap completed! Generated {result['num_images']} images")
        print("Parameters used:", result['parameters'])
    else:
        print("Error:", result['error'])
    
    # Example face swap with file upload
    result = api.face_swap_files(
        target_image_path="examples/poses/target.png",
        reference_image_path="examples/sess_9m7a6q7b7.jpg",
        output_dir="./results",
        steps=4,
        mask_strength=0.8,
        num_images=2
    )
    
    if result['success']:
        print("File upload face swap completed!")
        print(f"Generated {result['num_images']} images")
        print("ZIP saved to:", result['zip_path'])
        print("Extracted files:", result['extracted_files'])
    else:
        print("Error:", result['error'])
