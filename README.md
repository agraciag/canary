# Canary Speech Recognition Project

Real-time speech recognition and translation using NVIDIA's Canary model.

## Features
- Real-time audio transcription
- Multi-language support (EN, DE, ES, FR)
- Speech-to-text translation
- Docker containerization

## Requirements
- NVIDIA GPU with CUDA support
- Docker and Docker Compose
- NVIDIA Container Toolkit

## Setup

1. Clone the repository:
```bash
git clone https://github.com/agraciag/canary.git
cd canary
```

2. Create and start the Docker container:
```bash
mkdir workspace
docker compose up -d
```

3. Run the transcription script:
```bash
docker exec -it nemo-canary python /workspace/rtc_canary.py
```

## Configuration
Copy `config.example.yaml` to `config.local.yaml` and adjust settings.

## License
This project is licensed under CC-BY-NC-4.0 - see LICENSE file for details.

## Acknowledgments
- NVIDIA NeMo Toolkit
- Canary-1B model