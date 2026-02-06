import os
import shutil
import kagglehub
from pathlib import Path
from loguru import logger

def download_ieee_cis(output_dir: Path) -> Path:
    """Download IEEE-CIS Fraud Detection dataset from Kaggle to project data directory."""
    logger.info("Downloading IEEE-CIS Fraud Detection dataset...")

    if not os.environ.get("KAGGLE_USERNAME"):
        logger.warning("KAGGLE_USERNAME not set. Please configure Kaggle credentials:")
        logger.info("  1. Install kaggle: pip install kagglehub")
        logger.info("  2. Set environment variables:")
        logger.info("     export KAGGLE_USERNAME=your_username")
        logger.info("     export KAGGLE_KEY=your_api_key")
        logger.info("  3. Or place kaggle.json in ~/.kaggle/kaggle.json")
        logger.info("")
        logger.info("Alternatively, download manually from:")
        logger.info("  https://www.kaggle.com/competitions/ieee-fraud-detection/data")
        raise ValueError("Kaggle credentials not configured")

    try:
        path = kagglehub.competition_download("ieee-fraud-detection", force_download=True)
        logger.info(f"Downloaded to: {path}")

        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(path, output_dir)
        logger.info(f"Copied to project data directory: {output_dir}")

        return output_dir
    except Exception as e:
        if "auth" in str(e).lower() or "unauthenticated" in str(e).lower():
            logger.error("Authentication failed. Check your Kaggle credentials.")
        logger.error(f"Download failed: {e}")
        raise

if __name__ == "__main__":
    from fraud_detection.config import DATA_DIR
    download_ieee_cis(DATA_DIR)

