import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

from services.database_service import DatabaseService
from services.database_pontual_service import DatabaseService as DatabaseServicePontual

from models.weather_data import WeatherData
from datetime import date
from services.utils import valid_date



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

def setup_argparse():
    parser = argparse.ArgumentParser(description='Processa arquivos CSV e insere dados no banco de dados.')
    parser.add_argument('--csv-dir', type=str, help='Diretório contendo os arquivos CSV')
    
    # Configurações do banco de dados
    parser.add_argument('--db-name', type=str, help='Nome do banco de dados')
    parser.add_argument('--db-user', type=str, help='Usuário do banco de dados')
    parser.add_argument('--db-password', type=str, help='Senha do banco de dados')
    parser.add_argument('--db-host', type=str, help='Host do banco de dados')
    parser.add_argument('--db-port', type=int, help='Porta do banco de dados')

    # Configurações do banco de dados no pontual
    parser.add_argument('--db-name-pontual', type=str, help='Nome do banco de dados no pontual')
    parser.add_argument('--db-user-pontual', type=str, help='Usuário do banco de dados no pontual')
    parser.add_argument('--db-password-pontual', type=str, help='Senha do banco de dados no pontual')
    parser.add_argument('--db-host-pontual', type=str, help='Host do banco de dados no pontual')
    parser.add_argument('--db-port-pontual', type=int, help='Porta do banco de dados no pontual')
    # parser.add_argument('--use-other-service', action='store_true', help='Usar outro DataService')
    
    parser.add_argument('--start-date', type=valid_date, help='Data de início')
    parser.add_argument('--end-date', type=valid_date, help='Data de término')
    parser.add_argument('--mode', type=str, choices=['inmet', 'vehiclerun'], default='inmet', help='Modo de execução')
    return parser.parse_args()


   

def main():
    args = setup_argparse()

    # Carrega as variáveis de ambiente do arquivo .env
    load_dotenv(override=True)

    path_to_csv_dir = args.csv_dir or os.getenv('PATH_TO_CSV_DIR')

    logger.info(f"path_to_csv_dir: {path_to_csv_dir}")

    # Executa o modo específico
    if args.mode == 'vehiclerun':
        print(f"args: {args}")
        # Configurações do banco de dados
        db_config = {
            'dbname': args.db_name_pontual or os.getenv('DB_NAME_PONTUAL'),
            'user': args.db_user_pontual or os.getenv('DB_USER_PONTUAL'),
            'password': args.db_password_pontual or os.getenv('DB_PASSWORD_PONTUAL'),
            'host': args.db_host_pontual or os.getenv('DB_HOST_PONTUAL'),
            'port': args.db_port_pontual or os.getenv('DB_PORT_PONTUAL')
        }
        print(f"db_config: {db_config}")
        
        db_service = DatabaseServicePontual(db_config, args.start_date, args.end_date)
        db_service.run()
        
    elif args.mode == 'inmet':
        # Configurações do banco de dados
        db_config = {
            'dbname': args.db_name or os.getenv('DB_NAME'),
            'user': args.db_user or os.getenv('DB_USER'),
            'password': args.db_password or os.getenv('DB_PASSWORD'),
            'host': args.db_host,
            'port': args.db_port
        }
        # Diretório contendo os arquivos CSV
        data_dir = Path(path_to_csv_dir)
        db_service = DatabaseService(db_config, data_dir)
        db_service.run()


    # Fecha a conexão com o banco de dados
    db_service.close()
    logger.info("Database connection closed.")

if __name__ == "__main__":
    main()
