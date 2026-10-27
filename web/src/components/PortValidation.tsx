import { useJson } from '../data'

interface Row {
  mean: number
  std: number
  landing_rate: number
  mean_steps: number
}
interface Validation {
  python: Record<string, Row | string>
  browser: Record<string, Row>
}

const LABEL: Record<string, string> = {
  heuristic: 'Heurística de Gymnasium',
  dqn: 'DQN (red principal)',
  dqn_nofuel: 'DQN sin costo de combustible*',
}

/** Python (Box2D) vs navegador (planck.js), 100 episodios cada uno. */
export function PortValidation() {
  const v = useJson<Validation>('lunar_validation.json')
  if (!v) return null
  const keys = Object.keys(v.browser).filter((k) => typeof v.python[k] === 'object')
  return (
    <details className="card mb-5 px-4 py-3 text-sm">
      <summary className="cursor-pointer font-semibold">¿El port es fiel? Python vs navegador, 100 episodios cada uno</summary>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[520px] text-left">
          <thead>
            <tr className="border-b border-[var(--line)]">
              <th scope="col" className="py-1.5">Controlador</th>
              <th scope="col">Gymnasium (Python)</th>
              <th scope="col">planck.js (navegador)</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => {
              const p = v.python[k] as Row
              const b = v.browser[k]
              return (
                <tr key={k} className="border-b border-[var(--line)]/50">
                  <th scope="row" className="py-1.5 font-medium">{LABEL[k] ?? k}</th>
                  <td>
                    {p.mean.toFixed(1)} ± {p.std.toFixed(1)} · {Math.round(p.landing_rate * 100)} % aterrizajes
                  </td>
                  <td>
                    {b.mean.toFixed(1)} ± {b.std.toFixed(1)} · {Math.round(b.landing_rate * 100)} % aterrizajes
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs opacity-80">
        Los terrenos y las semillas no coinciden entre ambos lados (son generadores aleatorios distintos), así que se comparan
        distribuciones: las diferencias están dentro del ruido de 100 episodios. *Evaluada con la recompensa sin costo con la
        que se entrenó. Se reproduce con <code>python export_policy.py</code> y <code>VALIDATE_LUNAR=1 npx vitest run lunar</code>.
      </p>
    </details>
  )
}
