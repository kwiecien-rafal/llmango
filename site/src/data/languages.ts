/** The languages a prompt or a schema is written in, and what each one is called. */

export const LANGUAGES = ["en", "pl", "ja"] as const;

export type Language = (typeof LANGUAGES)[number];

export const LANGUAGE_NAMES: Record<Language, string> = {
  en: "English",
  pl: "Polish",
  ja: "Japanese",
};
