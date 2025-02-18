## Setup and Usage

### Docker Setup
1. Create docker-compose.yml:
```yaml
services:
  nemo:
    container_name: nemo-canary
    image: nvcr.io/nvidia/nemo:24.12.01
    runtime: nvidia 
    working_dir: /workspace
    stdin_open: true
    tty: true
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    volumes:
      - ./workspace:/workspace
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1 
              capabilities: [gpu]
```

2. Create workspace directory and start container:
```bash
mkdir workspace
docker compose up -d
```

3. Run Python scripts by executing in container:
```bash
docker exec -it nemo-canary python /workspace/your_script.py
```

## How to Use this Model

The model is available for use in the NeMo toolkit [4], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

Next steps would be:

Create audio processing script in workspace directory
Test basic transcription
Create manifest for translation testing
Test different language pairs