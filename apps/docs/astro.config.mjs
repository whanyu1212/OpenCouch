import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://whanyu1212.github.io",
  base: "/OpenCouch",
  trailingSlash: "always",
  integrations: [
    starlight({
      title: "OpenCouch",
      description: "Engineering a guided self-help program for anxiety.",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/whanyu1212/OpenCouch",
        },
      ],
      sidebar: [
        { label: "Overview", slug: "index" },
        { label: "Quickstart", slug: "quickstart" },
        { label: "Architecture", slug: "architecture" },
        { label: "Safety", slug: "safety" },
        {
          label: "Evaluations",
          items: [
            { label: "Methodology", slug: "evaluations/methodology" },
            { label: "November 2026 results", slug: "evaluations/2026-11-05" },
          ],
        },
      ],
    }),
  ],
});
