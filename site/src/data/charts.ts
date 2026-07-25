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
  experiment_id: string;
  charts: Chart[];
};

const modules = import.meta.glob<ChartIndex>("../../public/charts/*/index.json", {
  eager: true,
  import: "default",
});

export const experiments: ChartIndex[] = Object.values(modules).sort((a, b) =>
  a.experiment_id.localeCompare(b.experiment_id),
);

export function experiment(id: string): ChartIndex {
  const found = experiments.find((entry) => entry.experiment_id === id);
  if (!found) {
    throw new Error(`No charts for experiment ${id}. Run 'llmango analyze' first.`);
  }
  return found;
}

export function chart(experimentId: string, file: string): Chart {
  const found = experiment(experimentId).charts.find((entry) => entry.file === file);
  if (!found) {
    throw new Error(`No chart ${file} in experiment ${experimentId}.`);
  }
  return found;
}

export function chartSrc(experimentId: string, file: string): string {
  return `/charts/${experimentId}/${file}`;
}

/** Everything in a cell beyond the plotted share, such as the error count. */
export function asides(cell: Cell): [string, number][] {
  const plotted = ["value", "count", "n"];
  return Object.entries(cell).filter(([key]) => !plotted.includes(key));
}
