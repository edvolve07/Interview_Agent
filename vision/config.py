import os


class VisionConfig:
    def __init__(self):
        self.voice_provider = os.getenv("VOICE_PROVIDER", "nvidia")
        self.voice_model = os.getenv(
            "VOICE_MODEL", "meta/llama-3.1-8b-instruct"
        )

        self.vision_provider = os.getenv("VISION_PROVIDER", "nvidia")
        self.vision_model = os.getenv(
            "VISION_MODEL", "meta/llama-3.2-11b-vision-instruct"
        )
        self.groq_vision_model = os.getenv(
            "GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview"
        )

        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")
        self.nvidia_base_url = os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")

        self.frame_interval = float(os.getenv("FRAME_INTERVAL", "2.0"))
        self.max_image_size = int(os.getenv("MAX_IMAGE_SIZE", "512"))
        self.image_format = os.getenv("IMAGE_FORMAT", "jpeg")
        self.image_quality = int(os.getenv("IMAGE_QUALITY", "70"))
        self.motion_threshold = float(
            os.getenv("MOTION_THRESHOLD", "0.15")
        )

        self.summary_interval = int(
            os.getenv("SUMMARY_INTERVAL", "3")
        )
        self.max_summaries = int(os.getenv("MAX_SUMMARIES", "10"))
