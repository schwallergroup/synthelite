""" Module with script to download public data
"""
import argparse
import os
import sys

import requests
import tqdm
from huggingface_hub import hf_hub_download
from synthelite.chem.mol import Molecule
import pandas as pd

FILES_TO_DOWNLOAD = {
    "policy_model_onnx": {
        "filename": "azf/uspto_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_model.onnx",
    },
    "template_file": {
        "filename": "azf/uspto_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_unique_templates.csv.gz",
    },
    "ringbreaker_model_onnx": {
        "filename": "azf/uspto_ringbreaker_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_ringbreaker_model.onnx",
    },
    "ringbreaker_templates": {
        "filename": "azf/uspto_ringbreaker_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_ringbreaker_unique_templates.csv.gz",
    },
    "filter_policy_onnx": {
        "filename": "azf/uspto_filter_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_filter_model.onnx",
    },
    # "zinc_stock": {
    #     "filename": "stocks/zinc_stock.hdf5",
    #     "url": "https://ndownloader.figshare.com/files/23086469",
    # },
    "eMolecule_stock": {
        # "local_dir": "stocks",
        "hf_remote_path": "stocks/eMolecule.csv",
    },
    "synthelite_template_file": {
        # "local_dir": "reaction_templates",
        "hf_remote_path": "reaction_templates/uspto_templates.text-embedding-ada-002.csv",
    },
}


def _get_inchi_key(smiles: str) -> str:
    mol = Molecule(smiles=smiles)
    return mol.inchi_key


def _convert_stock_to_azf_format(source_file: str, target_file: str) -> None:
    df = pd.read_csv(source_file)
    df["inchi_key"] = df["mol"].progress_apply(_get_inchi_key)
    df.to_csv(target_file, index=False)


def _download_file(url: str, filename: str) -> None:
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping download.")
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(
            total=total_size, desc=os.path.basename(filename), unit="B", unit_scale=True
        )
        with open(filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()

def _download_file_hf(remote_path: str, local_dir: str) -> None:
    filename = os.path.basename(remote_path)
    filepath = os.path.join(local_dir, filename)
    if os.path.exists(filepath):
        print(f"File {filepath} already exists, skipping download.")
        return

    os.makedirs(local_dir, exist_ok=True)
    path = hf_hub_download(
        repo_id="SchwallerGroup/synthelite",
        filename=remote_path,
        repo_type="dataset",
        local_dir=local_dir
    )

def main() -> None:
    """Entry-point for CLI"""
    parser = argparse.ArgumentParser("download_public_data")
    parser.add_argument(
        "path",
        default=".",
        help="the path to download the files",
    )
    path = parser.parse_args().path

    try:
        for filespec in FILES_TO_DOWNLOAD.values():
            if "url" in filespec:
                _download_file(filespec["url"], os.path.join(path, filespec["filename"]))
            elif "hf_remote_path" in filespec:
                _download_file_hf(filespec["hf_remote_path"], path)
    except requests.HTTPError as err:
        print(f"Download failed with message {str(err)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
