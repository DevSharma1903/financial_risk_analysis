"""Historical balanced amendment-label builder; not used by the Colab workflow."""
import sys


if __name__ == "__main__":
    print("This historical script downsamples filings and labels amendments as positives.")
    print("It does not identify confirmed restatements. Use src/09_prepare_colab_dataset.py for the academic proxy workflow.")
    sys.exit(2)
