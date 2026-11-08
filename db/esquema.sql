-- Esquema de SmartBuilding (Edificio San Carlos): las ocho tablas que llena cargar_csv.py.
-- Docker lo ejecuta automáticamente la primera vez que crea la base (docker compose up).

CREATE TABLE edificio (
    id_edificio  SERIAL PRIMARY KEY,
    nombre       VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE torre (
    id_torre     SERIAL PRIMARY KEY,
    id_edificio  INTEGER NOT NULL REFERENCES edificio (id_edificio),
    numero       INTEGER NOT NULL CHECK (numero > 0),
    UNIQUE (id_edificio, numero)
);

-- El número del apartamento se guarda como texto ('102'); el piso es su centena.
CREATE TABLE apartamento (
    id_apartamento  SERIAL PRIMARY KEY,
    id_torre        INTEGER NOT NULL REFERENCES torre (id_torre),
    numero          VARCHAR(10) NOT NULL,
    piso            INTEGER NOT NULL CHECK (piso > 0),
    UNIQUE (id_torre, numero)
);

CREATE TABLE persona (
    id_persona      SERIAL PRIMARY KEY,
    id_apartamento  INTEGER NOT NULL REFERENCES apartamento (id_apartamento),
    nombre          VARCHAR(120) NOT NULL,
    telefono        VARCHAR(20),
    email           VARCHAR(120),
    tipo            VARCHAR(20) NOT NULL CHECK (tipo IN ('propietario', 'residente'))
);

CREATE TABLE mascota (
    id_mascota      SERIAL PRIMARY KEY,
    id_apartamento  INTEGER NOT NULL REFERENCES apartamento (id_apartamento),
    tipo            VARCHAR(40) NOT NULL
);

CREATE TABLE casillero (
    id_casillero    SERIAL PRIMARY KEY,
    id_apartamento  INTEGER NOT NULL UNIQUE REFERENCES apartamento (id_apartamento),
    codigo          VARCHAR(10) NOT NULL UNIQUE
);

CREATE TABLE parqueadero (
    id_parqueadero  SERIAL PRIMARY KEY,
    id_apartamento  INTEGER NOT NULL REFERENCES apartamento (id_apartamento),
    codigo          VARCHAR(10) NOT NULL UNIQUE
);

CREATE TABLE vehiculo (
    id_vehiculo     SERIAL PRIMARY KEY,
    id_parqueadero  INTEGER NOT NULL UNIQUE REFERENCES parqueadero (id_parqueadero),
    placa           VARCHAR(10) NOT NULL UNIQUE
);
