# Docker Compose Setup for InstantID Face Swap

This project includes multiple Docker Compose configurations for different deployment scenarios.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose v2+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA support

## Available Configurations

### 1. Basic Setup (`docker-compose.yml`)

The default configuration for basic usage.

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

**Features:**
- Single InstantID service on port 8888
- GPU support with NVIDIA runtime
- Persistent volumes for uploads, results, checkpoints, and models
- Health checks
- Automatic restart

### 2. Development Setup (`docker-compose.dev.yml`)

Optimized for development with live code reloading.

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Access the application at http://localhost:8888
# Access Jupyter notebook at http://localhost:8889
```

**Features:**
- Live code reloading
- Jupyter notebook for experimentation
- Development environment variables
- Source code mounted as volume

### 3. Production Setup (`docker-compose.prod.yml`)

Full production setup with reverse proxy, caching, and monitoring.

```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# Access application at http://localhost
# Access Grafana at http://localhost:3000 (admin/admin123)
# Access Prometheus at http://localhost:9090
```

**Features:**
- Nginx reverse proxy with SSL support
- Redis for caching/session storage
- Prometheus and Grafana monitoring
- Production security headers
- Rate limiting
- Memory limits and resource management

## GPU Requirements

Ensure your system has:

1. **NVIDIA Docker Runtime** installed:
```bash
# Install nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. **Verify GPU access**:
```bash
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

## Model Downloads

The models will be automatically downloaded on first startup. You can also pre-download them:

```bash
# Create directories
mkdir -p checkpoints/ControlNetModel models/antelopev2

# Download models (this will happen automatically in Docker)
# Or manually download to speed up first startup
```

## Configuration

### Environment Variables

Key environment variables (already set in docker-compose):

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`
- `CUDA_LAUNCH_BLOCKING=1`
- `TORCH_USE_CUDA_DSA=1`
- `NVIDIA_VISIBLE_DEVICES=all`

### Volume Mounts

- `./uploads` - Input images
- `./results` - Generated images
- `./checkpoints` - Model checkpoints
- `./models` - Face analysis models

### Ports

- **8888** - Main application (basic setup)
- **80/443** - Nginx proxy (production)
- **3000** - Grafana (production)
- **9090** - Prometheus (production)
- **8889** - Jupyter (development)

## Usage Examples

### Basic Usage
```bash
# Start basic setup
docker-compose up -d

# Wait for models to download (check logs)
docker-compose logs -f instantid-faceswap

# Access web interface
open http://localhost:8888
```

### Development
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Make code changes - they'll be reflected immediately
# Access Jupyter for experimentation
open http://localhost:8889
```

### Production Deployment
```bash
# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# Check all services are healthy
docker-compose -f docker-compose.prod.yml ps

# View aggregated logs
docker-compose -f docker-compose.prod.yml logs -f
```

## SSL Setup for Production

1. Obtain SSL certificates and place them in `nginx/ssl/`:
   - `cert.pem`
   - `key.pem`

2. Update `nginx/nginx.conf` to enable HTTPS (uncomment SSL lines)

3. Restart the nginx service:
```bash
docker-compose -f docker-compose.prod.yml restart nginx
```

## Monitoring

The production setup includes Prometheus and Grafana:

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

Import dashboards for:
- System metrics
- Docker metrics
- Application metrics

## Troubleshooting

### Common Issues

1. **GPU not detected**:
```bash
# Check NVIDIA runtime
docker info | grep nvidia

# Verify GPU in container
docker-compose exec instantid-faceswap nvidia-smi
```

2. **Out of memory errors**:
   - Reduce batch size
   - Check `PYTORCH_CUDA_ALLOC_CONF` settings
   - Monitor GPU memory usage

3. **Model download failures**:
   - Check internet connection
   - Verify Hugging Face access
   - Check disk space

4. **Permission issues**:
```bash
# Fix volume permissions
sudo chown -R $USER:$USER uploads results checkpoints models
```

### Logs

```bash
# View specific service logs
docker-compose logs -f instantid-faceswap

# View all logs
docker-compose logs -f

# Follow logs with timestamps
docker-compose logs -f -t
```

### Health Checks

```bash
# Check application health
curl http://localhost:8888/health

# Check all services status
docker-compose ps
```

## Scaling

For high-load scenarios, you can scale the application:

```bash
# Scale to multiple instances (requires load balancer)
docker-compose -f docker-compose.prod.yml up -d --scale instantid-faceswap=3
```

Note: For true horizontal scaling, you'll need to:
1. Use external storage for models
2. Implement proper session management
3. Configure load balancing
4. Use external Redis/database

## Performance Tuning

### GPU Memory Optimization
- Adjust `max_split_size_mb` in CUDA allocation config
- Use `enable_vae_slicing()` and `enable_xformers_memory_efficient_attention()`
- Monitor memory usage with `nvidia-smi`

### CPU and Memory
- Adjust Docker memory limits in compose files
- Tune worker processes for your CPU count
- Monitor with Grafana dashboards

## Security Considerations

### Production Security
- Change default passwords
- Use proper SSL certificates
- Configure firewall rules
- Regular security updates
- Monitor access logs

### API Security
- Implement authentication if needed
- Rate limiting (configured in nginx)
- Input validation
- File type restrictions
