"""Disabled historical downloader. The Colab workflow uses local SEC datasets only."""
import sys

if __name__ == "__main__":
    print("This historical script downloads filing text and is not used by the Colab workflow.")
    print("Prepare the local SUB/NUM datasets with src/09_prepare_colab_dataset.py instead.")
    sys.exit(2)
