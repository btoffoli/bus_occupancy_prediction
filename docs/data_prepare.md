# Predicting the Occupancy Rate of Public Transportation Buses at Bus Stops

## 1 - Running enviroment to load weather data

### 1.1 - Fist step, do download interesting csv files in https://portal.inmet.gov.br/dadoshistoricos and extract files in folder data. Try to keep just files with you'll work

#

### 1.2 - Set .env file with your settings

cp .env.example .env

#

### 1.3 - Run enviroment to load weather data, postgres and pgadmin using docker and docker-compose

Running docker-compose -f docker-coompise.dev.yml up -d

#

### 1.4 - Load weather data to database, remenber you must set the configuration in .env file

#### 1.4.1 - You must to create an enviroment for python

python -m venv .venv

#### 1.4.2 - Load enviroment and install depedencies

source .venv/bin/activate

#### 1.4.5 - Run application to copy data from files to database

python main.py

##

### 1.5 - To visualize data throught pgadmin, you must to register a server in pgadmin setting host database and port 5432
