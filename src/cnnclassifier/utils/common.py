import os
from box.exceptions import BoxValueError
import yaml
from cnnclassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads yaml file and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): path to the YAML file

    Raises:
        ValueError: if YAML file is empty
        Exception: for other unexpected errors

    Returns:
        ConfigBox: Parsed YAML content as ConfigBox
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        logger.exception(f"Error reading YAML file at {path_to_yaml}")
        raise 


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories.

    Args:
        path_to_directories (list): List of directory paths to create
        verbose (bool, optional): Whether to log creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves data to a JSON file.

    Args:
        path (Path): Path to the JSON file
        data (dict): Data to save
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise ValueError("Unable to load or parse JSON file.")

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves binary data using joblib.

    Args:
        data (Any): The object to be saved
        path (Path): File path to save the binary
    """
    try:
        joblib.dump(data, path)
        logger.info(f"Binary file saved at: {path}")
    except Exception as e:
        logger.error(f"Failed to save binary file at {path}: {e}")
        raise

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads binary data using joblib.

    Args:
        path (Path): File path to load the binary data from

    Returns:
        Any: The loaded Python object
    """
    try:
        data = joblib.load(path)
        logger.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load binary file from {path}: {e}")
        raise
@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the size of a file in a human-readable format."""
    try:
        size_in_bytes = os.path.getsize(path)
        size_in_kb = size_in_bytes / 1024

        if size_in_kb < 1024:
            return f"{size_in_kb:.2f} KB"
        else:
            size_in_mb = size_in_kb / 1024
            return f"{size_in_mb:.2f} MB"
    except Exception as e:
        logger.error(f"Failed to get size for {path}: {e}")
        raise


@ensure_annotations
def decodeImage(imgstring: str, filename: Path) -> None:
    """Decode base64 image string and save to file."""
    try:
        imgdata = base64.b64decode(imgstring)
        with open(filename, "wb") as f:
            f.write(imgdata)
        logger.info(f"Image successfully decoded and saved to: {filename}")
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise

@ensure_annotations
def encodeImage(image_path: Path) -> str:
    """Encodes an image file to a base64 string.

    Args:
        image_path (Path): The path to the image file

    Returns:
        str: Base64-encoded string of the image
    """
    try:
        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")
        logger.info(f"Image successfully encoded from: {image_path}")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        raise
