from pathlib import Path
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import read_yaml,create_directories
CONFIG_FILE_PATH=Path("config/config.yaml")
PARAMS_FILE_PATH=Path("params.yaml")
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config=read_yaml(config_filepath)
        # self.params=read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            kaggle_data=config.kaggle_data,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir ,
            kaggle_path=config.kaggle_path
        )
        return data_ingestion_config
