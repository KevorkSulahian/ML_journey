"""
Download and prepare the MovieLens dataset.
"""
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    zip_path = data_dir / "ml-100k.zip"
    
    # Download dataset
    if not zip_path.exists():
        print(f"Downloading MovieLens dataset to {zip_path}")
        download_file(MOVIELENS_URL, zip_path)
    
    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print("Dataset ready!")

if __name__ == "__main__":
    main()