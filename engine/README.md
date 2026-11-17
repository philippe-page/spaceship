# Local engine

The backend of Spaceship IDE's local mode: a small HTTP server that runs an open model on this machine with [MLX](https://github.com/ml-explore/mlx) and steers it from inside, in place of the hosted models and prompts of flight mode. `npm run local`, from the repository root, starts it with [uv](https://docs.astral.sh/uv/) together with the page.

## Models

You pick the model in the page's launch dialog, and it downloads only then, once, to the Hugging Face cache, at a pinned revision:

- [LFM3 1.7B](https://huggingface.co/mlx-community/LFM3-1.7B-OptiQ-4bit) (1.4 GB, Liquid AI, [LFM Open License](https://huggingface.co/LiquidAI/LFM3-1.7B))
- [Qwen3.5 2B](https://huggingface.co/mlx-community/Qwen3.5-2B-4bit) (1.8 GB, Alibaba, Apache 2.0)

Both have 24 layers, 6 of them full attention between cheaper ones (LFM3's short convolutions and sliding-window attention, Qwen3.5's linear attention), and both gate each attention output with a sigmoid, which the readouts below use. Each model's steering vectors are found anew when it loads.

## How each interaction works

- **Future vector steering** adds a steering vector to the residual stream of the second quarter of the layers (6 to 11 of 24) while the continuation is generated. Each dimension's vector is found by contrastive activation addition: the model reads the same story openings under instructions for each pole (or option), and the mean difference of its layer outputs is the direction. The knob's angle picks the pole and its distance sets how hard the vector pushes; nothing about the knob is written into the prompt.
- **Attention lock** adds a bias to the attention scores of the boxed phrase's tokens, and more to its keyword's, in every full-attention layer, so the model literally attends more to them while it writes.
- **Focus monitor** runs the model a few tokens ahead and boxes the words its full-attention layers attend to most, with each head weighted by how open its output gate is.
- **Influence lines and keyword relevance** are read from the same attention weights between sentences and keywords.
- **The fan menu** offers a fixed catalog of eight dimensions (tension, tone, pace and so on), plus the story's characters, picked out by capitalization, as a focus dimension; it ranks them by how far a push along each one moves the model's next-token distribution at the knob.
- **Realization** asks whether a sentence is likelier under a control's steering vector than under its opposite. It is the roughest of these readouts on models this small: it can misread a calm sentence as tense, and on Qwen3.5 2B it barely tells tension apart at all, though it reads tone.
- **Realtime regeneration** reads the story so far without steering, so moving a knob only changes the tokens after it.

## The inside view

To show that this is the model's internals and not a prompt, the **inside** view (on by default in local mode, toggled in the tray) runs an unsteered copy of the model beside the steered one, reading the same text and sampling from the same random draws. A word steering flipped, one the unsteered copy would not have written there, gets a dotted underline, brighter the more steering raised its probability. The sidebar shows, at the main knob, the candidates for the next word with and without steering, as the odds each word was drawn from (after the repetition penalty, temperature and top-p), and lists the latest flips, with a count of them all; hovering a generated word shows the same for it. With no knob set and no box locked, the two copies agree on every word.

## Platforms and settings

On an Apple Silicon Mac MLX runs on the GPU. On Windows or Linux it uses CUDA 12 when `nvidia-smi` finds an NVIDIA GPU, and otherwise the CPU, which is too slow to keep up with the knob; set `SPACESHIP_BACKEND` to `cuda` or `cpu` to choose. The CUDA backend has been tried on Windows 10 with an RTX 3080, where LFM3 1.7B writes about 11 tokens a second with the inside view on; Linux and the CPU backend have not been tried on a real machine yet. Nothing from NVIDIA needs installing beyond the driver: the CUDA libraries and headers come as Python packages.

On Windows, if Controlled Folder Access is on and the clone is under a protected folder such as Documents, allow `uv` through it, or the engine cannot create its environment.

The engine listens on `http://localhost:8765` and answers only pages served from this machine, so the published demo cannot use it; set `SPACESHIP_ENGINE_PORT` to change the port, and the page's `local-engine-url` local storage entry to match.

## Third-party code and weights

`lfm3.py`, the MLX implementation of LFM3, is adapted from [mlx-optiq](https://pypi.org/project/mlx-optiq/) 0.8.3 (MIT, © 2026 Thin Signal; the changes are listed in its header); Qwen3.5 runs on mlx-lm's own implementation. The weights are Liquid AI's and Alibaba's, under the licenses on their model pages.
