/**
 * The chart index written by `llmango analyze`, read at build time.
 *
 * Every number the site shows comes from here. The pipeline owns the numbers
 * and the drawings; this module only looks them up, so the site can never
 * disagree with the chart it is sitting next to.
 */

export type Cell = {
  value: number;
  count: number;
  n: number;
  [aside: string]: number;
};

export type Row = {
  label: string;
  cells: Cell[];
};

export type Chart = {
  metric: string;
  question_id: string | null;
  file: string;
  title: string;
  row_label: string;
  arms: string[];
  columns: string[];
  rows: Row[];
};

export type ChartIndex = {
  question_id: string;
  charts: Chart[];
};

const modules = import.meta.glob<ChartIndex>("../../public/charts/*/index.json", {
  eager: true,
  import: "default",
});

export const questions: ChartIndex[] = Object.values(modules).sort((a, b) =>
  a.question_id.localeCompare(b.question_id),
);

export function question(id: string): ChartIndex {
  const found = questions.find((entry) => entry.question_id === id);
  if (!found) {
    throw new Error(`No charts for question ${id}. Run 'llmango analyze' first.`);
  }
  return found;
}

export function chart(questionId: string, file: string): Chart {
  const found = question(questionId).charts.find((entry) => entry.file === file);
  if (!found) {
    throw new Error(`No chart ${file} for question ${questionId}.`);
  }
  return found;
}

export function chartSrc(questionId: string, file: string): string {
  return `/charts/${questionId}/${file}`;
}

/** Everything in a cell beyond the plotted share, such as the error count. */
export function asides(cell: Cell): [string, number][] {
  const plotted = ["value", "count", "n"];
  return Object.entries(cell).filter(([key]) => !plotted.includes(key));
}
