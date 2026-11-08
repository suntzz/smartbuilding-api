import { useState } from 'react'

function App() {
  const [residentes, setResidentes] = useState([])
  const [cargando, setCargando] = useState(false)
  const [error, setError] = useState('')
  const [consultado, setConsultado] = useState(false)

  async function consultar(ruta) {
    setCargando(true)
    setError('')
    setResidentes([])
    setConsultado(true)

    try {
      const respuesta = await fetch(`/api${ruta}`)

      if (!respuesta.ok) {
        if (respuesta.status === 404) {
          throw new Error('No se encontró el residente solicitado.')
        }

        throw new Error(
          `No se pudo completar la consulta. Código: ${respuesta.status}`
        )
      }

      const datos = await respuesta.json()

      // Una búsqueda devuelve un objeto.
      // La consulta de todos devuelve una lista.
      const lista = Array.isArray(datos)
        ? datos
        : datos === null
          ? []
          : [datos]

      if (lista.some((dato) => typeof dato !== 'object' || dato === null)) {
        throw new Error('La API devolvió un formato de datos inesperado.')
      }

      setResidentes(lista)
    } catch (err) {
      setError(
        err instanceof TypeError
          ? 'No se pudo conectar con la API. Revisa que FastAPI esté activo.'
          : err.message
      )
    } finally {
      setCargando(false)
    }
  }

  function buscarResidente(evento) {
    evento.preventDefault()

    const formulario = new FormData(evento.currentTarget)
    const id = String(formulario.get('residenteId') ?? '').trim()

    if (!/^[1-9]\d*$/.test(id)) {
      setError('Ingresa un ID entero mayor que cero.')
      return
    }

    consultar(`/residentes/${encodeURIComponent(id)}`)
  }

  function verTodos() {
    consultar('/residentes')
  }

  function mostrarValor(valor) {
    if (valor === null || valor === undefined) return '—'
    if (typeof valor === 'object') return JSON.stringify(valor)
    return String(valor)
  }

  const columnas = [
    ...new Set(residentes.flatMap((residente) => Object.keys(residente))),
  ]

  return (
    <>
      <header>
        <a className="logo" href="/">
          Edificio <span>San Carlos</span>
        </a>
        <span>Administración</span>
      </header>

      <main>
        <section className="hero">
          <h1>Residentes</h1>
          <p>
            Consulta la información de quienes viven en el edificio.
          </p>
        </section>

        <form onSubmit={buscarResidente}>
          <h2>Buscar residente</h2>

          <p className="subtitulo">
            Ingresa el ID de un residente o consulta la lista completa.
          </p>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="residente-id">
                ID del residente
              </label>

              <input
                id="residente-id"
                name="residenteId"
                type="number"
                min="1"
                step="1"
                placeholder="Ejemplo: 10"
                required
              />
            </div>

            <button type="submit" disabled={cargando}>
              Buscar
            </button>

            <button
              type="button"
              className="btn-secundario"
              onClick={verTodos}
              disabled={cargando}
            >
              Ver todos
            </button>
          </div>
        </form>

        <section
          className="card"
          aria-label="Resultados de la consulta"
          aria-busy={cargando}
        >
          <h2>Resultados</h2>

          <div aria-live="polite">
            {cargando && (
              <p className="vacio">Consultando residentes...</p>
            )}

            {error && (
              <p className="alerta alerta-error" role="alert">
                {error}
              </p>
            )}

            {!cargando && !error && residentes.length === 0 && (
              <p className="vacio">
                {consultado
                  ? 'No se encontraron residentes.'
                  : 'Los resultados de la consulta aparecerán aquí.'}
              </p>
            )}

            {!cargando && !error && residentes.length > 0 && (
              <>
                <p className="subtitulo">
                  Registros encontrados: {residentes.length}
                </p>

                <div className="tabla-wrapper">
                  <table>
                    <thead>
                      <tr>
                        {columnas.map((columna) => (
                          <th key={columna} scope="col">
                            {columna.replaceAll('_', ' ')}
                          </th>
                        ))}
                      </tr>
                    </thead>

                    <tbody>
                      {residentes.map((residente, indice) => (
                        <tr key={indice}>
                          {columnas.map((columna) => (
                            <td key={columna}>
                              {mostrarValor(residente[columna])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        </section>
      </main>

      <footer>
        SmartBuilding · Desarrollo de Software
      </footer>
    </>
  )
}

export default App