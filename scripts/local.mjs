import { spawn, spawnSync } from "node:child_process"
import { resolve } from "node:path"
import { setTimeout as wait } from "node:timers/promises"
import { createServer } from "vite"

/**
 * Local mode in one command: `npm run local` starts the MLX engine (`engine/`, run with uv) and, once it listens, the dev server,
 * and opens the page in local mode. MLX runs on Metal on a Mac; elsewhere it takes its CUDA backend when `nvidia-smi` finds an
 * NVIDIA GPU, else its CPU one, unless SPACESHIP_BACKEND says cuda or cpu.
 */
const ENGINE_DIR = resolve(import.meta.dirname, "../engine")
const ENGINE_HEALTH = `http://localhost:${process.env.SPACESHIP_ENGINE_PORT ?? 8765}/health`

const backend = process.platform === "darwin" ? undefined : (process.env.SPACESHIP_BACKEND ?? (hasNvidiaGpu() ? "cuda" : "cpu"))
const engine = spawn("uv", ["run", ...(backend ? ["--extra", backend] : []), "server.py"], { cwd: ENGINE_DIR, stdio: "inherit" })
engine.on("error", (error) => {
  console.error(error.code === "ENOENT" ? "Local mode runs the engine with uv: https://docs.astral.sh/uv/" : error.message)
  process.exit(1)
})
engine.on("exit", (code) => process.exit(code ?? 0))
// uv leaves an interrupt to the terminal, which sends it to the engine too; a stop sent to this process alone has to be passed on.
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => engine.kill("SIGTERM"))

// The first run installs the engine's packages before it listens, which takes a while; the page would find no engine until then.
while (!(await isEngineUp())) await wait(500)
const server = await createServer({ server: { open: "?local=true" } })
await server.listen()
server.printUrls()

function hasNvidiaGpu() {
  return spawnSync("nvidia-smi", ["-L"]).status === 0
}

function isEngineUp() {
  return fetch(ENGINE_HEALTH).then(
    (response) => response.ok,
    () => false,
  )
}
