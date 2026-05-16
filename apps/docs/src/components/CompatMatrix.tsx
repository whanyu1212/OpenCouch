import React, { useState } from 'react';
import styles from './PromptVisuals.module.css';

const approaches = [
  'motivational_interviewing',
  'cbt',
  'act',
  'dbt_skills',
  'grief_support',
  'interpersonal_therapy',
  'pfa',
];
const approachLabels = ['MI', 'CBT', 'ACT', 'DBT', 'Grief', 'IPT', 'PFA'];
const approachFull = [
  'Motivational Interviewing',
  'Cognitive Behavioural Therapy',
  'Acceptance and Commitment Therapy',
  'DBT skills',
  'Grief Support',
  'Interpersonal Therapy',
  'Psychological First Aid',
];

const rows: { responseStyle: string; desc: string; compat: boolean[] }[] = [
  { responseStyle: 'supportive',      desc: 'Default support lane — validate, reflect, and respond based on venting, strengths, or gentle guidance.', compat: [true, true, true, true, true, true, true] },
  { responseStyle: 'reflective',      desc: 'Explore a named recurring pattern carefully without over-analyzing.',                                      compat: [true, true, true, false, true, true, false] },
  { responseStyle: 'clarifying',      desc: 'Ask for the minimum context needed before choosing a deeper response.',                                    compat: [true, true, true, true, true, true, true] },
  { responseStyle: 'psychoeducation', desc: 'Explain one likely mind-body process in simple, non-diagnostic language.',                                compat: [true, true, true, true, true, true, true] },
  { responseStyle: 'technique',       desc: 'Structured therapeutic work without launching a named multi-turn exercise.',                              compat: [false, true, true, true, true, true, true] },
  { responseStyle: 'guided_exercise', desc: 'State-tracked practice such as grounding, breathing, thought work, values, or emotion regulation.',        compat: [false, true, true, true, false, false, true] },
  { responseStyle: 'closing',         desc: 'Wind down, reflect the useful part, and avoid opening a large new thread.',                               compat: [true, true, true, true, true, true, true] },
  { responseStyle: 'safety_check',    desc: 'Ambiguous risk signal — ask one direct safety clarification.',                                             compat: [false, false, false, false, false, false, true] },
  { responseStyle: 'crisis_response', desc: 'Confirmed risk — prioritize immediate safety and surface resources.',                                     compat: [false, false, false, false, false, false, true] },
];

export default function CompatMatrix() {
  const [hoveredCol, setHoveredCol] = useState<number | null>(null);

  return (
    <div className={styles.matrixRoot}>
      <div className={styles.matrixWrap}>
        <table className={styles.matrixTable}>
          <thead>
            <tr>
              <th className={styles.matrixCorner}>Response</th>
              <th className={styles.matrixDescCorner}>What it does</th>
              {approachLabels.map((label, i) => (
                <th
                  key={approaches[i]}
                  className={[styles.matrixModHead, hoveredCol === i ? styles.matrixModHeadActive : ''].join(' ')}
                  title={approachFull[i]}
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.responseStyle}>
                <td className={styles.matrixResponseStyleCell}>{row.responseStyle}</td>
                <td className={styles.matrixDescCell}>{row.desc}</td>
                {row.compat.map((allowed, i) => (
                  <td
                    key={approaches[i]}
                    className={[styles.matrixCell, hoveredCol === i ? styles.matrixCellHighlit : ''].join(' ')}
                    onMouseEnter={() => setHoveredCol(i)}
                    onMouseLeave={() => setHoveredCol(null)}
                  >
                    {allowed ? (
                      <span className={styles.matrixYes} title={`${row.responseStyle} + ${approachFull[i]}`}>✓</span>
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
        This is the intended routing surface, not a hard static registry.
        The dispatcher prompt in <code>agent/therapeutic/dispatch/prompt.py</code>
        selects <code>response_style</code> and <code>therapeutic_approach</code>;
        prompt source loading lives in <code>agent/therapeutic/prompting/sources.py</code>.
        Hover a column to highlight it.
      </p>
    </div>
  );
}
