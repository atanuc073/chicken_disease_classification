import os
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import get_size

import os
import zipfile
class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config

    def download_file(self):
        os.environ['KAGGLE_CONFIG_DIR'] =self.config.kaggle_path
       
        dataset_slug = self.config.kaggle_data
        destination_folder = self.config.local_data_file
        data_name=destination_folder+dataset_slug.split('/')[-1]+".zip"
        print(not os.path.exists(data_name))
        if not os.path.exists(data_name):
            # Create the command to download the dataset into the specified folder

            command = f"kaggle datasets download -d {dataset_slug} -p {destination_folder}"
            print(command)
            # Run the download command
            os.system(command)
            logger.info(f"{dataset_slug} downloaded ! ")
        else:
            logger.info(f"The file is already exists of size : {get_size(data_name)}")
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        dataset_slug = self.config.kaggle_data
        path_dir=unzip_path+dataset_slug.split('/')[-1]+".zip"
        os.makedirs(unzip_path, exist_ok=True)
        try:
            with zipfile.ZipFile(path_dir, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"file has been successfully exxtracted .")
        except:
            logger.info("there is some error while extracting the files")


