import gdown
from pathlib import Path

def download_synthetic_dataset():
    # sythetic dataset uploaded by authors
    # https://drive.google.com/drive/folders/1XuCyPFTM2Ro9kfCP_tWOF1uwBB-Hn3LH

    # url = "https://drive.google.com/drive/folders/1XuCyPFTM2Ro9kfCP_tWOF1uwBB-Hn3LH"
    url = "https://drive.google.com/drive/folders/1ip1q3zKL2N3-pubdOBnk0dw9AnnQSNpE"
    
    # OCR.npy included
    # url = "https://drive.google.com/drive/folders/1b4t-ZjyGOHLK-NJdOukFSdYVvZdWfv5m"

    # mkdir if it doesnt exist
    data_path = Path(__file__).resolve().parents[2] / "data/synthetic"
    print(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(url, output=data_path.parent)

if __name__ == "__main__":
    download_synthetic_dataset()
