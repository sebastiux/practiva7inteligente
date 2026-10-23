import { useEffect } from 'react'
import { Deferred } from './components/Layout'
import { useJson, type DqnData, type TabularData } from './data'
import {
  ConclusionsSection, DqnCurveSection, DqnTheorySection, FuelSection, ScalabilitySection, type SeedRow,
} from './sections/Dqn'
import {
  ConvergenceSection, EquationsSection, HeatmapSection, IntroSection, MdpSection, PseudocodeSection, RoboticsSection,
} from './sections/Tabular'

export default function Body({ tab, dqn, onReady }: { tab: TabularData | null; dqn: DqnData | null; onReady: () => void }) {
  const nofuel = useJson<DqnData>('dqn_nofuel.json')
  const seeds = useJson<SeedRow[]>('dqn_seeds.json')
  useEffect(onReady, [onReady])
  return (
    <>
      <Deferred id="intro" minHeight={700}>
        <IntroSection />
      </Deferred>
      <Deferred id="mdp" minHeight={1100}>
        <MdpSection data={tab} />
      </Deferred>
      <Deferred id="ecuaciones" minHeight={1100}>
        <EquationsSection data={tab} />
      </Deferred>
      <Deferred id="pseudocodigo" minHeight={1700}>
        <PseudocodeSection />
      </Deferred>
      <Deferred id="convergencia" minHeight={1500}>
        <ConvergenceSection data={tab} />
      </Deferred>
      <Deferred id="heatmaps" minHeight={2000}>
        <HeatmapSection data={tab} />
      </Deferred>
      <Deferred id="robotica" minHeight={500}>
        <RoboticsSection />
      </Deferred>
      <Deferred id="dqn-teoria" minHeight={1300}>
        <DqnTheorySection dqn={dqn} />
      </Deferred>
      <Deferred id="dqn-curva" minHeight={2000}>
        <DqnCurveSection dqn={dqn} seeds={seeds} />
      </Deferred>
      <Deferred id="dqn-combustible" minHeight={1800}>
        <FuelSection dqn={dqn} nofuel={nofuel} />
      </Deferred>
      <Deferred id="escalabilidad" minHeight={1200}>
        <ScalabilitySection dqn={dqn} />
      </Deferred>
      <Deferred id="conclusiones" minHeight={700}>
        <ConclusionsSection />
      </Deferred>
      <footer className="py-10 text-sm opacity-80">
        <p>
          Código fuente:{' '}
          <a className="underline" href="https://github.com/sebastiux/practiva7inteligente">
            github.com/sebastiux/practiva7inteligente
          </a>
          . Vite + React + TypeScript + Tailwind, Plotly.js y KaTeX. El águila es una ilustración original. Sitio sin
          afiliación con ningún club ni marca registrada.
        </p>
      </footer>
    </>
  )
}
