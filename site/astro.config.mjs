import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

// The charts under public/charts are written by `llmango analyze` and served
// verbatim, so no build step ever touches them.
export default defineConfig({
  site: "https://llmango.rafalkwiecien.com",
  integrations: [mdx()],
});
