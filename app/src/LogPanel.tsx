/** A collapsed-by-default progress log -- the raw "[analyse] ..." lines are
 * useful when something's wrong, but shouldn't take up screen space (or
 * grow the page) while everything's going fine. */
export function LogPanel({ lines }: { lines: string[] }) {
  if (lines.length === 0) return null;
  return (
    <details className="log-panel">
      <summary>Journal ({lines.length})</summary>
      <pre className="log">{lines.join("\n")}</pre>
    </details>
  );
}
