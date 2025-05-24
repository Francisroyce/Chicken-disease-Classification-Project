
import os
import time
import hashlib
import zipfile
from pathlib import Path
from urllib import request
from tenacity import retry, stop_after_attempt, wait_fixed
import zipfile
from zipfile import BadZipFile
from urllib import request
from cnnclassifier import logger
from cnnclassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        file_path = self.config.local_data_file
        url = self.config.source_URL

        if os.path.exists(file_path):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    corrupt = zip_ref.testzip()
                    if corrupt:
                        raise BadZipFile(f"Corrupted file in zip: {corrupt}")
                logger.info(f"File already exists and is a valid zip: {file_path}")
                return
            except BadZipFile as e:
                logger.warning(f"Corrupted zip file found. Deleting: {file_path}. Reason: {e}")
                os.remove(file_path)

        try:
            filename, headers = request.urlretrieve(url=url, filename=file_path)
            logger.info(f"{filename} downloaded successfully with headers: \n{headers}")
        except Exception as e:
            logger.error(f"Failed to download file from {url}. Error: {e}")
            raise

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
                logger.info(f"Extracted {self.config.local_data_file} to {unzip_path}")
        except BadZipFile as e:
            logger.error(f"Failed to extract zip file. BadZipFile: {e}")
            raise


