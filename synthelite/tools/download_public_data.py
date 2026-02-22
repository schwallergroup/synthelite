""" Module with script to download public data
"""
import argparse
import os
import sys

import requests
import tqdm

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
    "zinc_stock": {
        "filename": "stocks/zinc_stock.hdf5",
        "url": "https://ndownloader.figshare.com/files/23086469",
    },
    "eMolecule_stock": {
        "filename": "stocks/eMolecule.csv",
        "url": "https://zenodo.org/records/17883640/files/eMolecule.csv",
    },
    "synthelite_template_file": {
        "filename": "synthelite/uspto_templates.text-embedding-ada-002.csv",
        "url": "https://zenodo.org/records/17883640/files/uspto_templates.text-embedding-ada-002.csv",
    },
}

YAML_TEMPLATE = """expansion:
  uspto:
    - {}
    - {}
  ringbreaker:
    - {}
    - {}
filter:
  uspto: {}
stock:
  zinc: {}
"""


def _get_inchi_key(smiles: str) -> str:
    mol = Molecule(smiles=smiles)
    return mol.inchi_key


def _convert_stock_to_azf_format(source_file: str, target_file: str) -> None:
    df = pd.read_csv(source_file)
    df["inchi_key"] = df["mol"].progress_apply(_get_inchi_key)
    df.to_csv(target_file, index=False)


def _download_file(url: str, filename: str) -> None:
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
            _download_file(filespec["url"], os.path.join(path, filespec["filename"]))
    except requests.HTTPError as err:
        print(f"Download failed with message {str(err)}")
        sys.exit(1)

    # with open(os.path.join(path, "config.yml"), "w") as fileobj:
    #     path = os.path.abspath(path)
    #     fileobj.write(
    #         YAML_TEMPLATE.format(
    #             os.path.join(path, FILES_TO_DOWNLOAD["policy_model_onnx"]["filename"]),
    #             os.path.join(path, FILES_TO_DOWNLOAD["template_file"]["filename"]),
    #             os.path.join(
    #                 path, FILES_TO_DOWNLOAD["ringbreaker_model_onnx"]["filename"]
    #             ),
    #             os.path.join(
    #                 path, FILES_TO_DOWNLOAD["ringbreaker_templates"]["filename"]
    #             ),
    #             os.path.join(path, FILES_TO_DOWNLOAD["filter_policy_onnx"]["filename"]),
    #             os.path.join(path, FILES_TO_DOWNLOAD["stock"]["filename"]),
    #         )
    #     )
    # print("Configuration file written to config.yml")


if __name__ == "__main__":
    main()
