import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

// The charts under public/charts are written by `llmango analyze` and served verbatim.
export default defineConfig({
  site: "https://llmango.rafalkwiecien.com",
  markdown: {
    remarkRehype: {
      footnoteLabel: "footnotes",
      footnoteBackLabel: (index) => `back to reference ${index + 1}`,
    },
  },
  integrations: [mdx()],
});
