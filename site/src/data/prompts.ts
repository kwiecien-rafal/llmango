/** The prompt files the runs were sent from, read at build time rather than copied. */

import { LANGUAGES, LANGUAGE_NAMES, type Language } from "./languages";

export type Prompt = {
  lang: Language;
  name: string;
  text: string;
};

const modules = import.meta.glob<string>("../../../prompts/*/*/*.md", {
  eager: true,
  query: "?raw",
  import: "default",
});

const files = new Map(
  Object.entries(modules).map(([path, text]) => [
    path.split("/").slice(-3).join("/"),
    text,
  ]),
);

/** One question's prompt in one language, or a build failure naming the file. */
export function prompt(
  experiment: string,
  question: string,
  lang: Language,
): Prompt {
  const text = files.get(`${experiment}/${question}/${lang}.md`);
  if (text === undefined) {
    throw new Error(
      `No ${lang} prompt for question ${question} in experiment ${experiment}.`,
    );
  }
  return { lang, name: LANGUAGE_NAMES[lang], text: text.trimEnd() };
}

/** One question's prompt in every language, ordered as the charts order series. */
export function prompts(experiment: string, question: string): Prompt[] {
  return LANGUAGES.map((lang) => prompt(experiment, question, lang));
}
