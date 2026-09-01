/** Who writes the articles, and where each one links out to. */

export type Author = {
  name: string;
  url: string;
  sameAs: string[];
};

export const AUTHORS = {
  rk: {
    name: "Rafał Kwiecień",
    url: "https://rafalkwiecien.com",
    sameAs: ["https://github.com/kwiecien-rafal", "https://www.linkedin.com/in/rkwiecien"],
  },
} satisfies Record<string, Author>;

export type AuthorId = keyof typeof AUTHORS;
