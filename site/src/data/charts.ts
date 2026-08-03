/**
 * The chart index written by `llmango analyze`, read at build time.
 *
 * Every number the site shows comes from here. The pipeline owns the numbers
 * and the drawings; this module only looks them up, so the site can never
 * disagree with the chart it is sitting next to.
 *
 * A chart is looked up by the experiment that declares it and the name it is
 * declared under, so a page cites a chart the same way the experiment names it.
 */

export type Cell = {
  value: number;
  n: number;
  written: string;
  count?: number;
  lo?: number;
  hi?: number;
  written_interval?: string;
  [aside: string]: number | string | undefined;
};

export type Row = {
  label: string;
  cells: Cell[];
};

export type Chart = {
  name: string;
  number: string;
  file: string;
  questions: string[];
  title: string;
  row_label: string;
  unit: string;
  columns: string[];
  rows: Row[];
};

export type ChartIndex = {
  experiment: string;
  charts: Chart[];
};

const modules = import.meta.glob<ChartIndex>("../../public/charts/*/index.json", {
  eager: true,
  import: "default",
});

const indexes: ChartIndex[] = Object.values(modules);

/** One declared chart, or a build failure naming what the page asked for. */
export function chart(experiment: string, name: string): Chart {
  const index = indexes.find((entry) => entry.experiment === experiment);
  if (!index) {
    throw new Error(
      `No charts for experiment ${experiment}. Run 'llmango analyze' first.`,
    );
  }
  const found = index.charts.find((entry) => entry.name === name);
  if (!found) {
    throw new Error(`No chart ${name} in experiment ${experiment}.`);
  }
  return found;
}

export function chartSrc(experiment: string, file: string): string {
  return `/charts/${experiment}/${file}`;
}

/** Everything in a cell beyond the plotted number and the interval around it. */
export function asides(cell: Cell): [string, number][] {
  const plotted = ["value", "count", "n", "lo", "hi", "written", "written_interval"];
  return Object.entries(cell).filter(
    (entry): entry is [string, number] =>
      !plotted.includes(entry[0]) && typeof entry[1] === "number",
  );
}
