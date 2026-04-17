import React from 'react';
import styles from './TerminalWindow.module.css';

type TerminalWindowProps = {
  title?: string;
  children: React.ReactNode;
};

export default function TerminalWindow({
  title = 'bash — ~/OpenCouch',
  children,
}: TerminalWindowProps): JSX.Element {
  return (
    <div className={styles.terminal}>
      <div className={styles.titleBar}>
        <div className={styles.dots} aria-hidden="true">
          <span className={`${styles.dot} ${styles.dotRed}`} />
          <span className={`${styles.dot} ${styles.dotYellow}`} />
          <span className={`${styles.dot} ${styles.dotGreen}`} />
        </div>
        <div className={styles.titleText}>{title}</div>
        <div className={styles.titleSpacer} aria-hidden="true" />
      </div>
      <pre className={styles.body}>
        <code>{children}</code>
      </pre>
    </div>
  );
}
