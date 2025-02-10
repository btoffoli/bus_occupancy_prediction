-- migrations/001_create_weather_table.sql
CREATE TABLE weather_data (
    id BIGSERIAL,
    city VARCHAR(255) NOT NULL,
    reading_time TIMESTAMPTZ NOT NULL,
    precipitation NUMERIC,
    station_pressure NUMERIC,
    max_pressure NUMERIC,
    min_pressure NUMERIC,
    global_radiation NUMERIC,
    temperature NUMERIC,
    dew_point NUMERIC,
    max_temperature NUMERIC,
    min_temperature NUMERIC,
    max_dew_point NUMERIC,
    min_dew_point NUMERIC,
    max_humidity NUMERIC,
    min_humidity NUMERIC,
    humidity NUMERIC,
    wind_direction NUMERIC,
    wind_gust NUMERIC,
    wind_speed NUMERIC,
    PRIMARY KEY (id, reading_time)
) PARTITION BY RANGE (reading_time);

ALTER TABLE weather_data ADD CONSTRAINT unique_weather_data_reading_city UNIQUE (reading_time, city);

CREATE OR REPLACE FUNCTION create_weather_data_partition(month_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_name := 'weather_data_' || to_char(month_date, 'YYYY_MM');
    start_date := date_trunc('month', month_date);
    end_date := start_date + INTERVAL '1 month';

    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I PARTITION OF weather_data
        FOR VALUES FROM (%L) TO (%L);
    ', partition_name, start_date, end_date);

    EXECUTE format('
        CREATE INDEX IF NOT EXISTS %I ON %I (reading_time);
    ', 'idx_reading_time_' || to_char(month_date, 'YYYY_MM'), partition_name);
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    current DATE := '2022-01-01'::DATE;
    end_date DATE := '2026-01-01'; -- Até o final de 2025
BEGIN
    WHILE current < end_date LOOP
        PERFORM create_weather_data_partition(current);
        current := current + INTERVAL '1 month';
    END LOOP;
END $$;



-- verificando se foram criadas
SELECT relname AS partition_name
FROM pg_class
WHERE relkind = 'r' AND relname LIKE 'weather_data_%'
ORDER BY relname;