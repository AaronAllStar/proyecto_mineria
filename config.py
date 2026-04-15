import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    DATASET_PATH = os.getenv("DATASET_PATH", "full_dataset_thesis - 1.csv")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/dino_model.joblib")
    ENCODER_PATH = os.getenv("ENCODER_PATH", "models/encoders.joblib")
    REPORT_OUTPUT = os.getenv("REPORT_OUTPUT", "reports/")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

    @classmethod
    def ensure_dirs(cls):
        """Ensure necessary directories exist."""
        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)
        os.makedirs(cls.REPORT_OUTPUT, exist_ok=True)

Config.ensure_dirs()
