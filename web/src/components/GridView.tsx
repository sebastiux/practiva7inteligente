import { ACTION_DELTAS } from '../rl/gridworld'

export interface GridViewProps {
  map: string[]
  policy?: number[][]
  values?: number[][]
  /** Celdas a sombrear con opacidad (p.ej. visitas normalizadas 0..1) */
  mask?: number[][]
  paths?: { cells: [number, number][]; color: string; label: string; dashed?: boolean }[]
  highlight?: [number, number][]
  annotations?: { r: number; c: number; text: string; color?: string }[]
  title: string
  className?: string
}

const C = 24

/** Mapa del Paso del Contrabandista en SVG: obstáculos, A, B, flechas de política y rutas. */
export function GridView({ map, policy, values, mask, paths, highlight, annotations, title, className = '' }: GridViewProps) {
  const R = map.length
  const W = map[0].length
  let vmin = Infinity
  let vmax = -Infinity
  if (values) {
    values.forEach((row, i) =>
      row.forEach((v, j) => {
        if (map[i][j] === '#') return
        vmin = Math.min(vmin, v)
        vmax = Math.max(vmax, v)
      }),
    )
  }
  const valueColor = (v: number) => {
    const t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin || 1)))
    // azul marino -> amarillo
    const a = [11, 31, 58]
    const b = [255, 210, 0]
    const mix = a.map((x, k) => Math.round(x + (b[k] - x) * t))
    return `rgb(${mix.join(',')})`
  }

  return (
    <svg
      viewBox={`-2 -2 ${W * C + 4} ${R * C + 4}`}
      className={`w-full h-auto select-none ${className}`}
      role="img"
      aria-label={title}
    >
      <title>{title}</title>
      <rect x={-2} y={-2} width={W * C + 4} height={R * C + 4} rx={6} fill="#0d2a1c" />
      {map.map((row, i) =>
        [...row].map((ch, j) => {
          const x = j * C
          const y = i * C
          const obstacle = ch === '#'
          const fill = obstacle
            ? '#0B1F3A'
            : values
              ? valueColor(values[i][j])
              : (i + j) % 2
                ? '#1d5a35'
                : '#226840'
          return (
            <g key={`${i}-${j}`}>
              <rect x={x} y={y} width={C} height={C} fill={fill} stroke="rgba(255,255,255,0.09)" strokeWidth={0.6} />
              {obstacle && (
                <path d={`M${x + 4} ${y + 4} L${x + C - 4} ${y + C - 4} M${x + C - 4} ${y + 4} L${x + 4} ${y + C - 4}`} stroke="#FFD200" strokeWidth={1.4} opacity={0.55} />
              )}
              {mask && !obstacle && (
                <rect x={x} y={y} width={C} height={C} fill="#FFD200" opacity={Math.min(0.85, mask[i][j])} />
              )}
            </g>
          )
        }),
      )}
      {highlight?.map(([r, c]) => (
        <rect key={`h${r}-${c}`} x={c * C + 1.5} y={r * C + 1.5} width={C - 3} height={C - 3} fill="none" stroke="#ff5a5a" strokeWidth={2.5} strokeDasharray="4 2" rx={3} />
      ))}
      {policy &&
        map.map((row, i) =>
          [...row].map((ch, j) => {
            if (ch === '#' || ch === 'B') return null
            const a = policy[i][j]
            const [dr, dc] = ACTION_DELTAS[a]
            const cx = j * C + C / 2
            const cy = i * C + C / 2
            const len = 7.5
            const n = Math.hypot(dr, dc)
            const x2 = cx + (dc / n) * len
            const y2 = cy + (dr / n) * len
            const x1 = cx - (dc / n) * len * 0.7
            const y1 = cy - (dr / n) * len * 0.7
            const ang = Math.atan2(dr, dc)
            const h1 = [x2 - 4.5 * Math.cos(ang - 0.5), y2 - 4.5 * Math.sin(ang - 0.5)]
            const h2 = [x2 - 4.5 * Math.cos(ang + 0.5), y2 - 4.5 * Math.sin(ang + 0.5)]
            const stroke = values ? '#ffffff' : '#FFF6D5'
            return (
              <g key={`p${i}-${j}`} stroke={stroke} strokeWidth={1.6} strokeLinecap="round" opacity={0.92}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} />
                <polyline points={`${h1.join(',')} ${x2},${y2} ${h2.join(',')}`} fill="none" />
              </g>
            )
          }),
        )}
      {paths?.map((p) => (
        <polyline
          key={p.label}
          points={p.cells.map(([r, c]) => `${c * C + C / 2},${r * C + C / 2}`).join(' ')}
          fill="none"
          stroke={p.color}
          strokeWidth={3.2}
          strokeLinejoin="round"
          strokeLinecap="round"
          strokeDasharray={p.dashed ? '7 5' : undefined}
          opacity={0.95}
        />
      ))}
      {map.map((row, i) =>
        [...row].map((ch, j) =>
          ch === 'A' || ch === 'B' ? (
            <g key={`m${i}-${j}`}>
              <circle cx={j * C + C / 2} cy={i * C + C / 2} r={C / 2 - 1} fill={ch === 'A' ? '#FFF6D5' : '#FFD200'} stroke="#0B1F3A" strokeWidth={2} />
              <text x={j * C + C / 2} y={i * C + C / 2 + 5} textAnchor="middle" fontSize={14} fontWeight={800} fill="#0B1F3A">
                {ch}
              </text>
            </g>
          ) : null,
        ),
      )}
      {annotations?.map((a) => (
        <text
          key={a.text}
          x={a.c * C}
          y={a.r * C}
          fontSize={13}
          fontWeight={700}
          fill={a.color ?? '#FFF6D5'}
          stroke="#0B1F3A"
          strokeWidth={3}
          paintOrder="stroke"
        >
          {a.text}
        </text>
      ))}
    </svg>
  )
}
