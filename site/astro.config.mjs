import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

/** Prose links off the site open in a new tab; the layout's own links set this themselves. */
function rehypeExternalLinks() {
  const open = (node) => {
    if (node.tagName === "a" && /^https?:\/\//.test(node.properties?.href ?? "")) {
      node.properties.target = "_blank";
      node.properties.rel = ["noopener"];
    }
    for (const child of node.children ?? []) open(child);
  };
  return open;
}

// The charts under public/charts are written by `llmango analyze` and served verbatim.
export default defineConfig({
  site: "https://llmango.rafalkwiecien.com",
  markdown: {
    remarkRehype: {
      footnoteLabel: "footnotes",
      footnoteBackLabel: (index) => `back to reference ${index + 1}`,
    },
    rehypePlugins: [rehypeExternalLinks],
  },
  integrations: [mdx()],
});
