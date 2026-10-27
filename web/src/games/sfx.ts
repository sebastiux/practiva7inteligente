// Efectos de sonido sintetizados con WebAudio (sin archivos). Se activan sólo
// tras un gesto del usuario (botón START) y respetan el silencio guardado.

let ctx: AudioContext | null = null
let master: GainNode | null = null
let muted = (() => {
  try {
    return localStorage.getItem('sfx-muted') === '1'
  } catch {
    return false
  }
})()

export function isMuted() {
  return muted
}

export function setMuted(m: boolean) {
  muted = m
  try {
    localStorage.setItem('sfx-muted', m ? '1' : '0')
  } catch {
    /* sin almacenamiento */
  }
  if (master) master.gain.value = m ? 0 : 0.5
}

/** Llamar desde un handler de clic/tecla. */
export function unlockAudio() {
  if (!ctx) {
    const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AC) return
    ctx = new AC()
    master = ctx.createGain()
    master.gain.value = muted ? 0 : 0.5
    master.connect(ctx.destination)
  }
  if (ctx.state === 'suspended') void ctx.resume()
}

function noiseBuffer(seconds: number) {
  const c = ctx!
  const buf = c.createBuffer(1, Math.floor(c.sampleRate * seconds), c.sampleRate)
  const d = buf.getChannelData(0)
  let last = 0
  for (let i = 0; i < d.length; i++) {
    // ruido café (más grave que el blanco)
    last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02
    d[i] = last * 3.5
  }
  return buf
}

function env(g: GainNode, t: number, a: number, peak: number, dur: number) {
  g.gain.setValueAtTime(0.0001, t)
  g.gain.exponentialRampToValueAtTime(peak, t + a)
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur)
}

function ready() {
  return ctx && master && !muted
}

export function whistle(long = false) {
  if (!ready()) return
  const c = ctx!
  const t = c.currentTime
  const n = long ? 3 : 1
  for (let k = 0; k < n; k++) {
    const t0 = t + k * 0.28
    const o = c.createOscillator()
    const lfo = c.createOscillator()
    const lg = c.createGain()
    const g = c.createGain()
    o.type = 'sine'
    o.frequency.value = 2900
    lfo.frequency.value = 38
    lg.gain.value = 140
    lfo.connect(lg).connect(o.frequency)
    env(g, t0, 0.01, 0.18, k === n - 1 ? 0.55 : 0.22)
    o.connect(g).connect(master!)
    o.start(t0)
    lfo.start(t0)
    o.stop(t0 + 0.6)
    lfo.stop(t0 + 0.6)
  }
}

export function cheer() {
  if (!ready()) return
  const c = ctx!
  const t = c.currentTime
  const src = c.createBufferSource()
  src.buffer = noiseBuffer(2.2)
  const bp = c.createBiquadFilter()
  bp.type = 'bandpass'
  bp.frequency.value = 900
  bp.Q.value = 0.6
  const g = c.createGain()
  env(g, t, 0.25, 0.9, 2.1)
  src.connect(bp).connect(g).connect(master!)
  src.start(t)
}

export function thud() {
  if (!ready()) return
  const c = ctx!
  const t = c.currentTime
  const o = c.createOscillator()
  const g = c.createGain()
  o.type = 'sine'
  o.frequency.setValueAtTime(160, t)
  o.frequency.exponentialRampToValueAtTime(40, t + 0.3)
  env(g, t, 0.005, 0.8, 0.35)
  o.connect(g).connect(master!)
  o.start(t)
  o.stop(t + 0.4)
}

export function blip(freq = 660, dur = 0.08) {
  if (!ready()) return
  const c = ctx!
  const t = c.currentTime
  const o = c.createOscillator()
  const g = c.createGain()
  o.type = 'square'
  o.frequency.value = freq
  env(g, t, 0.005, 0.06, dur)
  o.connect(g).connect(master!)
  o.start(t)
  o.stop(t + dur + 0.02)
}

export function chime() {
  if (!ready()) return
  ;[523, 659, 784, 1047].forEach((f, i) => setTimeout(() => blip(f, 0.25), i * 110))
}

export function explosion() {
  if (!ready()) return
  const c = ctx!
  const t = c.currentTime
  const src = c.createBufferSource()
  src.buffer = noiseBuffer(1.4)
  const lp = c.createBiquadFilter()
  lp.type = 'lowpass'
  lp.frequency.setValueAtTime(1800, t)
  lp.frequency.exponentialRampToValueAtTime(120, t + 1.2)
  const g = c.createGain()
  env(g, t, 0.01, 1.4, 1.3)
  src.connect(lp).connect(g).connect(master!)
  src.start(t)
  thud()
}

/** Rumor continuo del motor: devuelve un control para subir/bajar el volumen. */
export function engineLoop() {
  if (!ctx || !master) return null
  const c = ctx
  const src = c.createBufferSource()
  src.buffer = noiseBuffer(2)
  src.loop = true
  const lp = c.createBiquadFilter()
  lp.type = 'lowpass'
  lp.frequency.value = 500
  const g = c.createGain()
  g.gain.value = 0
  src.connect(lp).connect(g).connect(master)
  src.start()
  return {
    set(level: number) {
      g.gain.setTargetAtTime(muted ? 0 : level, c.currentTime, 0.03)
    },
    stop() {
      try {
        src.stop()
      } catch {
        /* ya detenido */
      }
    },
  }
}
