# Guía: SmartBuilding en la máquina virtual Debian

```
Navegador (React + Vite)  →  FastAPI (Python)  →  PostgreSQL (Docker)
      puerto 5173                puerto 8000           puerto 5432
```

Todo se hace dentro de Debian, en una terminal. Copia cada bloque tal cual.
Donde dice `sudo`, Debian te pedirá tu contraseña de usuario.

---

## 0. En tu Mac: Extension Pack de VirtualBox (opcional)

El profe pasó el *Extension Pack* 7.2.14. Solo sirve si tu VirtualBox también es la 7.2.14
(se ve en VirtualBox → Acerca de). Se descarga de https://download.virtualbox.org/virtualbox/7.2.14/
(el archivo `Oracle_VirtualBox_Extension_Pack-7.2.14-174565.vbox-extpack`) y se abre con doble clic.
No hace falta para nada de lo que sigue.

## 1. Traer los cambios del repo

```bash
cd ~/smartbuilding-api
git pull
```

Si tu carpeta tiene otro nombre o está en otra ruta, cambia la primera línea.

## 2. Instalar Docker (una sola vez)

```bash
sudo apt update
sudo apt install -y docker.io docker-compose python3-venv git curl
sudo usermod -aG docker $USER
```

Después **cierra sesión en Debian y vuelve a entrar** (o reinicia la máquina virtual) para que tu usuario pueda usar Docker sin `sudo`.

Comprueba que funciona:

```bash
docker run --rm hello-world
```

## 3. Crear el archivo .env

```bash
cd ~/smartbuilding-api
cp .env.example .env
nano .env
```

Cambia `cambia-esta-clave` por una contraseña tuya. Guarda con `Ctrl+O`, `Enter` y sal con `Ctrl+X`.
El `.env` no se sube a GitHub.

## 4. Encender PostgreSQL

```bash
docker compose up -d
```

> Si responde `'compose' is not a docker command`, tu Debian trae la versión antigua: usa `docker-compose up -d` (con guion) aquí y en todos los pasos.

La primera vez descarga PostgreSQL 16 y crea las 8 tablas (`db/esquema.sql`). Revisa que estén:

```bash
docker compose exec db psql -U smartbuilding -d smartbuilding -c '\dt'
```

Deben salir: apartamento, casillero, edificio, mascota, parqueadero, persona, torre y vehiculo.

## 5. Preparar Python y cargar los residentes

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cargar_csv.py
```

Debe terminar con `Carga completada y confirmada en PostgreSQL` y `persona: 128`.
Si lo ejecutas otra vez, se niega (las tablas ya tienen datos): es lo esperado.

Pruebas del profe:

```bash
python -m pytest tests/test_residentes.py -v
```

Deben pasar las 4.

## 6. Terminal 1: encender FastAPI

```bash
cd ~/smartbuilding-api
source .venv/bin/activate
uvicorn app.main:app --reload
```

Deja esta terminal abierta. Puedes probar en el navegador de Debian: http://127.0.0.1:8000/residentes

## 7. Instalar Node.js con nvm (una sola vez)

Abre **otra terminal** y ejecuta el comando que pasó el profe:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.7/install.sh | bash
```

Cierra esa terminal y abre una nueva (para que reconozca `nvm`), y luego:

```bash
nvm install --lts
node -v
npm -v
```

`node -v` debe mostrar v22 o superior.

## 8. Terminal 2: encender la página

```bash
cd ~/smartbuilding-api/frontend
npm install
npm run dev
```

Abre en el navegador de Debian **http://localhost:5173**, pulsa **Ver todos** (salen los 128 residentes) o busca por ID (por ejemplo, 10).

Para generar los archivos de publicación (la carpeta `dist`):

```bash
npm run build
```

---

## Cada vez que vuelvas a trabajar

```bash
cd ~/smartbuilding-api && docker compose up -d
source .venv/bin/activate && uvicorn app.main:app --reload
```

y en otra terminal:

```bash
cd ~/smartbuilding-api/frontend && npm run dev
```

## Si algo falla

| Mensaje | Qué hacer |
|---|---|
| `permission denied ... docker.sock` | No cerraste sesión después del paso 2. Cierra sesión y vuelve a entrar |
| `port is already allocated` / `5432` en uso | Hay otro PostgreSQL en Debian: `sudo systemctl stop postgresql` y repite `docker compose up -d` |
| `password authentication failed` en `cargar_csv.py` | Cambiaste la clave del `.env` después de crear la base. Para empezar de cero (borra los datos): `docker compose down -v` y `docker compose up -d`, luego repite `python cargar_csv.py` |
| La página dice "No se pudo conectar con la API" | La terminal 1 (uvicorn) no está encendida |
| `nvm: command not found` | Cierra la terminal y abre una nueva después de instalar nvm |
| `connection refused` en el puerto 5432 | PostgreSQL está apagado: `docker compose up -d` |

## Qué hay en el repo

| Ruta | Qué es |
|---|---|
| `docker-compose.yml`, `db/esquema.sql` | PostgreSQL y las 8 tablas |
| `.env.example` | Plantilla de credenciales (se copia a `.env`) |
| `cargar_csv.py` | Carga inicial del CSV a las tablas (del profe) |
| `app/datos.py`, `app/main.py` | API de residentes contra PostgreSQL (del profe) |
| `tests/test_residentes.py` | Pruebas contra PostgreSQL (del profe) |
| `frontend/` | Página en React + Vite (`App.jsx` y `styles.css` del profe) |
| `app/api_vision.py`, `requirements-vision.txt` | La parte de cámaras, opcional. No hace falta instalarla en Debian |
