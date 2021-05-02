import gdown
from pathlib import Path

def download_synthetic_dataset():
    # sythetic dataset uploaded by authors
    # https://drive.google.com/drive/folders/1XuCyPFTM2Ro9kfCP_tWOF1uwBB-Hn3LH

    url = "https://drive.google.com/drive/folders/1XuCyPFTM2Ro9kfCP_tWOF1uwBB-Hn3LH"

    # mkdir if it doesnt exist
    data_path = Path("data/synthetic")
    data_path.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(url, output=data_path.parent)