import { lazy, Suspense, useMemo, type ReactNode } from 'react'
import type { TabularData } from '../data'
import { Playground } from './Playground'

const GridGame = lazy(() => import('../games/grid3d/GridGame'))
const LunarGame = lazy(() => import('../games/lunar/LunarGame'))

function hasWebGL() {
  try {
    const c = document.createElement('canvas')
    return !!(c.getContext('webgl2') || c.getContext('webgl'))
  } catch {
    return false
  }
}

function Loading() {
  return (
    <div className="grid h-[min(68vh,620px)] min-h-[380px] place-items-center rounded-2xl border border-[var(--line)] bg-navy-950 text-cream-100">
      <p className="font-display text-3xl tracking-widest text-gold-400">Cargando el estadio…</p>
    </div>
  )
}

/** Sección de juego a todo lo ancho, con estética de "arena". */
export function GameSection({ id, kicker, title, children }: { id: string; kicker: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={`${id}-h`} className="jersey-stripes -mx-5 border-b border-[var(--line)] px-5 py-14 md:py-20">
      <div className="mb-6 flex items-end gap-4">
        <span className="text-5xl leading-none md:text-6xl" aria-hidden>
          🎮
        </span>
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--accent)]">{kicker}</p>
          <h2 id={`${id}-h`} className="font-display text-4xl leading-tight tracking-wide md:text-5xl">
            {title}
          </h2>
        </div>
      </div>
      {children}
    </section>
  )
}

export function GridGameBlock({ tab }: { tab: TabularData | null }) {
  const gl = useMemo(hasWebGL, [])
  return (
    <GameSection id="juego-contrabandista" kicker="Juego 1 · SARSA vs Q-learning en 3D" title="El Paso del Contrabandista 3D">
      <div className="prose-rl mb-5 max-w-none">
        <p>
          La misma malla 20×20, ahora como cancha. La barrera es una fila de defensas, la brecha está marcada en rojo y la
          portería espera en B. Cada paso es una jugada: el robot <strong>patina</strong> si resbala, se lleva una{' '}
          <strong>tarjeta roja</strong> si choca y celebra el <strong>gol</strong> al llegar.
        </p>
        <ul>
          <li>
            <strong>Duelo:</strong> SARSA y Q-learning juegan a la vez con sus tablas Q finales de Python (semilla 0). Es la
            hipótesis del Entregable 2 en vivo: uno cruza la brecha y el otro la rodea.
          </li>
          <li>
            <strong>Entrena en vivo:</strong> un agente aprende desde cero. Usa la pretemporada o el modo turbo, mira el mapa de
            calor en el pasto y edita la cancha con clics.
          </li>
          <li>
            <strong>Juega tú:</strong> maneja el robot con el teclado o el D-pad, con el mismo resbalón, y compara tu promedio
            con el de los agentes.
          </li>
        </ul>
      </div>
      {gl ? (
        <Suspense fallback={<Loading />}>
          <GridGame tab={tab} />
        </Suspense>
      ) : (
        <div className="card p-4">
          <p className="mb-3 text-sm">Tu navegador no tiene WebGL; aquí está la versión 2D del entrenador interactivo.</p>
          <Playground />
        </div>
      )}
    </GameSection>
  )
}

export function LunarGameBlock({ children }: { children?: ReactNode }) {
  const gl = useMemo(hasWebGL, [])
  return (
    <GameSection id="juego-alunizaje" kicker="Juego 2 · Deep Q-Learning en 3D" title="Alunizaje DQN">
      <div className="prose-rl mb-5 max-w-none">
        <p>
          Portamos la física de <code>LunarLander</code> de Gymnasium a TypeScript con <strong>planck.js</strong> (el mismo
          motor Box2D). Usamos las mismas constantes, el mismo terreno aleatorio, la misma recompensa y la misma observación
          normalizada. Así, la <strong>red DQN entrenada en PyTorch</strong> (exportada a JSON, 18,180 pesos) pilota aquí
          mismo, a 50 Hz. Las barras muestran sus cuatro <code>Q(s, a)</code> en cada instante; la más alta es la acción que
          ejecuta. También puedes pilotar tú y ver qué haría la red en tu lugar.
        </p>
      </div>
      {children}
      {gl ? (
        <Suspense fallback={<Loading />}>
          <LunarGame />
        </Suspense>
      ) : (
        <p className="card p-4">Tu navegador no tiene WebGL, así que el juego 3D no puede mostrarse. Abajo está el video del aterrizaje.</p>
      )}
    </GameSection>
  )
}
