import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Bus Occupancy Prediction",
  description: "A System of Bus Occupancy Prediction using GPT networks",
  srcDir: ".", // Define a raiz do projeto como o diretório de origem
  // exclude: ['node_modules/**', 'dist/**'], // Ignora diretórios desnecessários
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "Home", link: "/" },
      // { text: "Examples", link: "/markdown-examples" },
    ],

    sidebar: [
      {
        text: "Fine tune",
        items: [
          { text: "How to make a Fine tune", link: "/docs/fine_tuning" },
          {
            text: "Fine tune with notebook",
            link: "/docs/fine_tuning_with_notebook",
          },
        ],
      },
      {
        text: "Predictions",
        items: [
          {
            text: "How to make a prediction without notebook",
            link: "/docs/fine_tuning#running-prediction-with-mistral-example",
          },
        ],
      },
      {
        text: "Article",
        items: [
          {
            text: "Article",
            link: "/docs/article.claude",
          },
        ],
      },
    ],

    socialLinks: [{ icon: "github", link: "https://github.com/btoffoli" }],
  },
});
