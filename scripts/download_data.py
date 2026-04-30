import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from multi_user_rag.config import (RAW_DATA_PATH)
import argparse
import requests
from tqdm import tqdm

def download_file(url: str, output_path: str) -> None:
    """
    Stream-download a file from a URL to disk.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download {url} | Status Code: {response.status_code}"
        )

    temp_path = output_path + ".part"

    total_size = int(response.headers.get("Content-Length", 0))
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()

    os.rename(temp_path, output_path)
    print(f"\nDownload complete: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download dataset from a URL."
    )
    parser.add_argument("--url", required=True, help="Dataset URL")
    parser.add_argument("--output", default=RAW_DATA_PATH, help=f"Output file path (default: {RAW_DATA_PATH})")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_file(args.url, args.output)
