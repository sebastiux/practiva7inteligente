/* Águila estilizada ORIGINAL (no es ningún escudo ni marca registrada):
   alas abiertas geométricas, cabeza de perfil y una "pelota" en las garras. */
export function Eagle({ className = '', title }: { className?: string; title?: string }) {
  return (
    <svg viewBox="0 0 240 200" className={className} role={title ? 'img' : undefined} aria-hidden={title ? undefined : true}>
      {title && <title>{title}</title>}
      <defs>
        <linearGradient id="eg-wing" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#FFE680" />
          <stop offset="1" stopColor="#FFD200" />
        </linearGradient>
      </defs>
      {/* ala izquierda: plumas escalonadas */}
      <path d="M112 78 L18 30 L40 52 L8 50 L38 70 L6 76 L42 88 L14 100 L54 104 L36 118 L84 112 Z" fill="url(#eg-wing)" />
      {/* ala derecha (espejo) */}
      <path d="M128 78 L222 30 L200 52 L232 50 L202 70 L234 76 L198 88 L226 100 L186 104 L204 118 L156 112 Z" fill="url(#eg-wing)" />
      {/* cuerpo */}
      <path d="M100 74 Q120 60 140 74 L146 120 Q120 150 94 120 Z" fill="#FFD200" />
      {/* pecho con franjas tipo jersey */}
      <path d="M106 92 L134 92 L130 104 L110 104 Z" fill="#0B1F3A" opacity="0.85" />
      <path d="M110 110 L130 110 L127 120 L113 120 Z" fill="#0B1F3A" opacity="0.85" />
      {/* cabeza y pico */}
      <circle cx="120" cy="58" r="17" fill="#FFF6D5" />
      <path d="M134 54 L152 60 L136 66 Z" fill="#FFD200" />
      <circle cx="126" cy="54" r="3" fill="#0B1F3A" />
      {/* cola */}
      <path d="M106 132 L120 172 L134 132 Z" fill="#FFE680" />
      {/* garras y balón */}
      <circle cx="120" cy="186" r="11" fill="#FFF6D5" stroke="#0B1F3A" strokeWidth="2" />
      <path d="M120 178 l5 4 -2 6 h-6 l-2 -6 z" fill="#0B1F3A" />
      <path d="M108 150 L104 176 M132 150 L136 176" stroke="#FFD200" strokeWidth="5" strokeLinecap="round" />
    </svg>
  )
}
