import os
import threading
import time
import sys
import gdown
import zipfile
from GANGetStarted import logger
from GANGetStarted.config.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.stop_animation = False
        self.current_message = ""
        self.animation_lock = threading.Lock()
    
    def animate(self):
        symbols = ['-', '\\', '|', '/']
        i = 0
        while not self.stop_animation:
            with self.animation_lock:
                message = self.current_message
            sys.stdout.write(f"\r{message} {symbols[i % len(symbols)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
    
    def download_data(self) -> str:
        try:
            data_url = self.config.source_url
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {data_url} into file {zip_download_dir}")

            file_id = data_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloaded data from {data_url} into file {zip_download_dir}")
        except Exception as e:
            raise e
        
    def unzip_data(self):
        try:
            unzip_path = self.config.unzip_dir
            zip_download_dir = self.config.local_data_file
            os.makedirs(unzip_path, exist_ok=True)
            logger.info(f"Unzipping data from {zip_download_dir} into file {unzip_path}")
            
            with zipfile.ZipFile(zip_download_dir, 'r') as zip_ref:
                total_files = len(zip_ref.infolist())
                start_time = time.time()
                extracted_files = 0

                # Start the animation thread
                self.stop_animation = False
                animation_thread = threading.Thread(target=self.animate)
                animation_thread.start()

                for file in zip_ref.infolist():
                    zip_ref.extract(file, unzip_path)
                    extracted_files += 1
                    elapsed_time = time.time() - start_time
                    remaining_files = total_files - extracted_files
                    estimated_total_time = elapsed_time / extracted_files * total_files
                    estimated_remaining_time = estimated_total_time - elapsed_time
                    minutes, seconds = divmod(estimated_remaining_time, 60)
                    time_remaining = f"{int(minutes)}m {int(seconds)}s"
                    
                    with self.animation_lock:
                        self.current_message = f"Unzipping data {' ' * (len(str(total_files)) - len(str(extracted_files)))}{extracted_files}/{total_files} - Estimated time remaining: {time_remaining}"
            
            # Stop the animation
            self.stop_animation = True
            animation_thread.join()

            logger.info(f"Unzipping completed data from {zip_download_dir} into file {unzip_path}")
        except Exception as e:
            self.stop_animation = True
            raise e