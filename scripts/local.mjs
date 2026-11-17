import { spawn, spawnSync } from "node:child_process"
import { homedir } from "node:os"
import { join, resolve } from "node:path"
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
const [uv, ...uvArgs] = findUv()
const engine = spawn(uv, [...uvArgs, "run", ...(backend ? ["--extra", backend] : []), "server.py"], { cwd: ENGINE_DIR, stdio: "inherit" })
engine.on("error", (error) => {
  console.error(error.message)
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

/** uv as a command line: from the PATH, from where its installer puts it (a terminal opened before the install does not have that on its PATH yet), or as a pip-installed module. */
function findUv() {
  const candidates = [["uv"], [join(homedir(), ".local", "bin", "uv")], ["python3", "-m", "uv"], ["python", "-m", "uv"], ["py", "-m", "uv"]]
  const found = candidates.find(([command, ...args]) => spawnSync(command, [...args, "--version"]).status === 0)
  if (found) return found
  const install =
    process.platform === "win32"
      ? 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
      : "curl -LsSf https://astral.sh/uv/install.sh | sh"
  console.error(`Local mode runs the engine with uv, which is not installed. Install it, then run this again:\n\n  ${install}\n`)
  process.exit(1)
}

function hasNvidiaGpu() {
  return spawnSync("nvidia-smi", ["-L"]).status === 0
}

function isEngineUp() {
  return fetch(ENGINE_HEALTH).then(
    (response) => response.ok,
    () => false,
  )
}
