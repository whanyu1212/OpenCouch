import React, { useState } from 'react';
import styles from './PromptVisuals.module.css';

const modalities = ['pfa', 'cbt', 'grief_support', 'act'];
const modalityLabels = ['PFA', 'CBT', 'Grief', 'ACT'];
const modalityFull = [
  'Psychological First Aid (+ DBT skills)',
  'Cognitive Behavioural Therapy',
  'Grief Support',
  'Acceptance and Commitment Therapy',
];

const rows: { mode: string; desc: string; compat: boolean[] }[] = [
  { mode: 'supportive_conversation', desc: 'Default support lane — validate, reflect, and respond based on venting, strengths, or gentle guidance. MI baseline.',  compat: [true,  true,  true,  true] },
  { mode: 'safety_check',           desc: 'Ambiguous risk signal — gentle clarification before routing.',                                                           compat: [true,  false, false, false] },
  { mode: 'crisis_response',        desc: 'Confirmed risk — prioritise safety, surface resources.',                                                                 compat: [true,  false, false, false] },
  { mode: 'orientation',            desc: 'New user — understand context and goals before full support. MI baseline.',                                              compat: [false, false, false, false] },
  { mode: 'pattern_reflection',     desc: 'Pattern reflection — help the user examine recurring themes carefully. MI baseline.',                                    compat: [false, true,  true,  true] },
  { mode: 'guided_exercise',        desc: 'Structured self-help — grounding, thought work, behavioural activation, or defusion.',                                  compat: [true,  true,  false, true] },
  { mode: 'psychoeducation',        desc: 'Explain one likely mind-body process in simple, non-diagnostic language.',                                               compat: [true,  true,  true,  true] },
  { mode: 'out_of_scope',           desc: 'Request outside boundaries — redirect with clear explanation.',                                                          compat: [false, false, false, false] },
  { mode: 'realignment',            desc: 'Session drift or rupture — acknowledge the miss and re-attune. MI baseline.',                                            compat: [false, false, false, false] },
];

export default function CompatMatrix() {
  const [hoveredCol, setHoveredCol] = useState<number | null>(null);

  return (
    <div className={styles.matrixRoot}>
      <div className={styles.matrixWrap}>
        <table className={styles.matrixTable}>
          <thead>
            <tr>
              <th className={styles.matrixCorner}>Mode</th>
              <th className={styles.matrixDescCorner}>What it does</th>
              {modalityLabels.map((label, i) => (
                <th
                  key={modalities[i]}
                  className={[styles.matrixModHead, hoveredCol === i ? styles.matrixModHeadActive : ''].join(' ')}
                  title={modalityFull[i]}
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.mode}>
                <td className={styles.matrixModeCell}>{row.mode}</td>
                <td className={styles.matrixDescCell}>{row.desc}</td>
                {row.compat.map((allowed, i) => (
                  <td
                    key={modalities[i]}
                    className={[styles.matrixCell, hoveredCol === i ? styles.matrixCellHighlit : ''].join(' ')}
                    onMouseEnter={() => setHoveredCol(i)}
                    onMouseLeave={() => setHoveredCol(null)}
                  >
                    {allowed ? (
                      <span className={styles.matrixYes} title={`${row.mode} + ${modalityFull[i]}`}>✓</span>
                    ) : (
                      <span className={styles.matrixNo}>—</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.matrixNote}>
        Rules enforced in <code>prompts/catalog.py</code> — invalid combinations raise at build time, not silently at runtime.
        Motivational Interviewing is applied as a baseline overlay (via <code>MODE_BASELINE_FILES</code>) to modes marked "MI baseline" above, not as a selectable modality.
        DBT skills are bundled into the PFA modality file set.
        Hover a column to highlight it. Hover a <span className={styles.matrixYes} style={{display:'inline-flex',width:20,height:20,fontSize:'0.8rem'}}>✓</span> for the exact pairing.
      </p>
    </div>
  );
}
