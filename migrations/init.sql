-- migrations/001_create_weather_table.sql
CREATE TABLE IF NOT EXISTS weather_data (
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



CREATE TABLE IF NOT EXISTS public.vehiclerun_bus_stop_occupation (
    itinerary_code character varying(255) COLLATE pg_catalog."default" NOT NULL,
    itinerary_size integer NOT NULL,
    scheduled_time timestamp without time zone NOT NULL,
    trip_start_time timestamp without time zone NOT NULL,
    trip_completion_time timestamp without time zone NOT NULL,
    vehicle_id integer NOT NULL,
    busstop_code character varying(255) COLLATE pg_catalog."default" NOT NULL,
    bustop_location integer NOT NULL,
    reading_time timestamp without time zone NOT NULL,
    occupation integer NOT NULL,
    occupation_geo geometry(Point,4326) NOT NULL,
    occupation_location integer NOT NULL,
    normalized_location double precision NOT NULL,

    temperature NUMERIC,
    humidity NUMERIC,
    wind_speed NUMERIC,    
    precipitation NUMERIC,   


    CONSTRAINT vehiclerunbusstopoccupation_pkey PRIMARY KEY (itinerary_code, scheduled_time, busstop_code)
) PARTITION BY RANGE (scheduled_time);


CREATE OR REPLACE FUNCTION create_vehiclerun_bus_stop_occupation_partition(start_date DATE, end_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    partition_range TEXT;
BEGIN
    -- Loop through each month in the given date range
    FOR start_date, end_date IN
        SELECT
            d,
            d + INTERVAL '1 month' - INTERVAL '1 day'
        FROM generate_series(start_date, end_date, '1 month') AS d
    LOOP
        -- Define the partition name and range
        partition_name := 'vehiclerun_bus_stop_occupation_' || TO_CHAR(start_date, 'YYYY_MM');
        partition_range := 'FOR VALUES FROM (''' || start_date || ''') TO (''' || end_date || ''')';

        -- Create the partition
        EXECUTE 'CREATE TABLE IF NOT EXISTS public.' || partition_name || ' PARTITION OF public.vehiclerun_bus_stop_occupation ' || partition_range;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

SELECT create_vehiclerun_bus_stop_occupation_partition('2022-01-01'::DATE, '2026-01-01'::DATE);

-- verificando se foram criadas
SELECT relname AS partition_name
FROM pg_class
WHERE relkind = 'r' AND 
    (relname LIKE 'weather_data_%'
        OR relname LIKE 'vehiclerun_bus_stop_occupation_%')
ORDER BY relname;



