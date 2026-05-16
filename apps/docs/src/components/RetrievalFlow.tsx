import React from 'react';
import s from './RetrievalFlow.module.css';

/* ================================================================
   RetrievalFlow — Hybrid RRF diagram

   Shows the two scorers converging into RRF fusion. Static (no
   interaction needed) — the visual itself tells the full story.
   ================================================================ */

export default function RetrievalFlow(): React.JSX.Element {
  return (
    <div className={s.root}>
      {/* Input */}
      <div className={s.inputNode}>
        <span className={s.inputIcon}>{'\u2709'}</span>
        <span>User message</span>
      </div>

      {/* Fork */}
      <div className={s.fork}>
        <div className={s.forkLine} />
        <div className={s.forkLine} />
      </div>

      {/* Two scorers */}
      <div className={s.scorers}>
        <div className={`${s.scorer} ${s.scorerToken}`}>
          <div className={s.scorerHeader}>
            <span className={s.scorerDot} />
            <span className={s.scorerLabel}>Token-recall</span>
          </div>
          <div className={s.scorerDetail}>
            Tokenize query (stopword-filtered). Score each record by
            <code>|query {'\u2229'} record| / |query|</code>. Keep {'\u2265'} 0.33.
          </div>
          <div className={s.scorerStrengths}>
            <span className={s.strengthGood}>Proper nouns</span>
            <span className={s.strengthGood}>Medication names</span>
            <span className={s.strengthGood}>Short queries</span>
          </div>
        </div>

        <div className={`${s.scorer} ${s.scorerEmbed}`}>
          <div className={s.scorerHeader}>
            <span className={s.scorerDot} />
            <span className={s.scorerLabel}>Embedding cosine</span>
          </div>
          <div className={s.scorerDetail}>
            Compute query embedding. Cosine similarity against stored
            embeddings. Keep {'\u2265'} 0.5.
          </div>
          <div className={s.scorerStrengths}>
            <span className={s.strengthGood}>Stemming</span>
            <span className={s.strengthGood}>Synonyms</span>
            <span className={s.strengthGood}>Paraphrase</span>
          </div>
        </div>
      </div>

      {/* Merge */}
      <div className={s.merge}>
        <div className={s.mergeLine} />
        <div className={s.mergeLine} />
      </div>

      {/* RRF fusion */}
      <div className={s.fusionNode}>
        <span className={s.fusionLabel}>RRF fusion</span>
        <code className={s.fusionFormula}>score = {'\u03A3'} 1/(k + rank), k=60</code>
      </div>

      {/* Output */}
      <div className={s.outputConn} />
      <div className={s.outputNode}>
        <span>Top-k {'\u2192'} WorkingMemoryEntry dicts</span>
      </div>
    </div>
  );
}
