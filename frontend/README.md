# Frontend de SmartBuilding

Página de residentes del Edificio San Carlos hecha con React y Vite. Consulta la API de FastAPI.

```bash
npm install      # la primera vez
npm run dev      # abre http://localhost:5173
npm run build    # genera la carpeta dist para publicar
```

FastAPI debe estar encendido en el puerto 8000 (`uvicorn app.main:app --reload` desde la raíz del repo).
Vite reenvía las peticiones `/api/...` a `http://127.0.0.1:8000/...`, así que no hace falta configurar CORS.
