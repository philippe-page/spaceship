import { readdirSync } from "node:fs"
import { resolve } from "node:path"
import { defaultClientConditions, defineConfig } from "vite"

export default defineConfig({
  base: "/spaceship/",
  // ONNX Runtime Web is taken in the build that leaves its 29 MB of WebAssembly out of the site (see src/engine/worker.ts).
  worker: { format: "es" },
  resolve: { conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions] },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        ...Object.fromEntries(
          readdirSync(resolve(import.meta.dirname, "extra"))
            .filter((file) => file.endsWith(".html"))
            .map((file) => [`extra/${file.slice(0, -".html".length)}`, resolve(import.meta.dirname, "extra", file)]),
        ),
      },
    },
  },
})
