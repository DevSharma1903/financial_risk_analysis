"""Legacy entry point intentionally disabled; see README.md for Colab workflow."""
import sys

if __name__ == "__main__":
    print("The legacy pipeline trains from stale cached features and amendment proxy labels.")
    print("Use `python src/09_prepare_colab_dataset.py`, then run the Colab GPU notebook.")
    sys.exit(2)
