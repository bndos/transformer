"""Script to download and extract the dataset."""

import os
import zipfile

import requests


def download_dataset(
    url: str, data_path: str = "data", filename: str = "fra.txt"
):
    """Download a dataset from a URL and extract it."""
    if os.path.exists(data_path + "/" + filename):
        print("Dataset already exists.")
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    filename_zip = url.split("/")[-1]
    response = requests.get(url)

    with open(filename_zip, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(filename_zip, "r") as zip_ref:
        zip_ref.extractall(filename)
