import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from services.file_reader import FileReader
from services.database_service import DatabaseService
from models.weather_data import WeatherData

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Carrega as variáveis de ambiente do arquivo .env
    load_dotenv(override=True)

    # Configurações do banco de dados
    db_config = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432))
    }

    db_config_pontual = {
        'dbname': os.getenv('DB_NAME_PONTUAL'),
        'user': os.getenv('DB_USER_PONTUAL'),
        'password': os.getenv('DB_PASSWORD_PONTUAL'),
        'host': os.getenv('DB_HOST_PONTUAL', 'localhost'),
        'port': int(os.getenv('DB_PORT_PONTUAL', 5432))

    }

    path_to_csv_dir = os.getenv('PATH_TO_CSV_DIR')

    logger.info(f"path_to_csv_dir: {path_to_csv_dir}")


    # Diretório contendo os arquivos CSV
    data_dir = Path(path_to_csv_dir)

    # Inicializa o serviço de banco de dados
    db_service = DatabaseService(db_config)

    # Processa cada arquivo CSV no diretório
    for csv_file in data_dir.glob('*.CSV'):
        logger.info(f"Processing file: {csv_file}")
        try:
            for weather_data_list in FileReader.read_csv(csv_file):
                # logger.info(f"weather_data_generator: {weather_data_generator}")
                # Insere os dados do arquivo CSV no banco de dados
                # for weather_data_list in weather_data_generator:
                logger.info(f"weather_data: {len(weather_data_list)}")
                db_service.insert_weather_data_batch(weather_data_list)
            logger.info(f"Successfully processed file: {csv_file}")
        except Exception as e:
            logger.error(f"Error processing file {csv_file}: {e}")

    # Fecha a conexão com o banco de dados
    db_service.close()
    logger.info("Database connection closed.")

if __name__ == "__main__":
    main()