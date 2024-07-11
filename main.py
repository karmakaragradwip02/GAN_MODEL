from GANGetStarted import logger
from GANGetStarted.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline

STAGE_NAME = "DATA INGESTION STAGE"

try: 
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e