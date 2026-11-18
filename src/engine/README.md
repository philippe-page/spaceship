# Local engine

The engine of Spaceship IDE's local mode: it runs an open model in the page, in a worker on WebGPU with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), and steers it from inside, in place of the hosted models and prompts of flight mode. There is nothing to install and no server: the page fetches the model from Hugging Face the first time it is picked, and the browser keeps it. The runtime's WebAssembly (29 MB) is not part of the site either: it comes from jsDelivr when a model loads.

## Models

You pick the model in the page's launch dialog, and it downloads only then, once, as its 4-bit ONNX export at a pinned revision:

- [LFM3 1.7B](https://huggingface.co/onnx-community/LFM3-1.7B-ONNX) (1.1 GB, Liquid AI, [LFM Open License](https://huggingface.co/LiquidAI/LFM3-1.7B))
- [Qwen3.5 2B](https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT) (1.4 GB, Alibaba, Apache 2.0)

Both have 24 layers, 6 of them full attention between cheaper ones (LFM3's short convolutions and sliding-window attention, Qwen3.5's linear attention), and both gate each attention output with a sigmoid, which the readouts below use. Each model's steering vectors are found anew each time it loads.

## How a model is opened up

An exported model is a fixed graph of operators with no way in. Its graph file is small, though (under 1 MB; the weights sit in a separate file), so the page rewires it as it loads (`onnx-graph.ts`, `instruments.ts`):

- an input per steered layer, added to that layer's output, and that output exposed, for finding the vectors;
- an input added to the attention scores of the full-attention layers;
- an output with those layers' attention weights. The attention operator keeps its weights to itself on WebGPU, so they are rebuilt from its queries and keys with plain operators;
- an output with the log-probability of a given token at each position;
- an output with how far apart the rows of a batch are at each layer, for the inside view.

## How each interaction works

- **Future vector steering** adds a steering vector to the residual stream of the second quarter of the layers (6 to 11 of 24) while the continuation is generated. Each dimension's vector is found by contrastive activation addition: the model reads the same story openings under instructions for each pole (or option), and the mean difference of its layer outputs is the direction. The knob's angle picks the pole and its distance sets how hard the vector pushes; nothing about the knob is written into the prompt.
- **Attention lock** adds a bias to the attention scores of the boxed phrase's tokens, and more to its keyword's, in every full-attention layer, so the model literally attends more to them while it writes. Measured, the keyword's share of attention rises four to nine times on both models. On LFM3 that brings the keyword back into the story (27 of 48 continuations mention it, against 7 unlocked); on Qwen3.5 it made no countable difference at this strength (2 of 48 against 4), and begins to at about four times the bias.
- **Focus monitor** runs the model a few tokens ahead and boxes the words its full-attention layers attend to most, with each head weighted by how open its output gate is.
- **Influence lines and keyword relevance** are read from the same attention weights between sentences and keywords.
- **The fan menu** offers a fixed catalog of eight dimensions (tension, tone, pace and so on), plus the story's characters, picked out by capitalization, as a focus dimension; it ranks them by how far a push along each one moves the model's next-token distribution at the knob.
- **Realization** asks whether a sentence is likelier under a control's steering vector than under its opposite. It is the roughest of these readouts on models this small: it can misread a calm sentence as tense.
- **Realtime regeneration** reads the story so far without steering, so moving a knob only changes the tokens after it.

## The inside view

To show that this is the model's internals and not a prompt, the **inside** view (on by default in local mode, toggled in the tray) runs an unsteered copy of the model beside the steered one, reading the same text and sampling from the same random draws. A word steering flipped, one the unsteered copy would not have written there, gets a dotted underline, brighter the more steering raised its probability. The sidebar shows, at the main knob, the model's layers from input (bottom) to output (top): a bar for how far each layer's state is from the unsteered copy's, as a share of that state's size, orange where a steering vector is added and marked where the attention lock biases attention, with the same drift over the last 48 tokens beside it. Below that it shows the candidates for the next word with and without steering, as the odds each word was drawn from (after the repetition penalty, temperature and top-p), and lists the latest flips, with a count of them all; hovering a generated word shows the same for it. With no knob set and no box locked, the two copies agree on every word.

## What it needs, and what has been tried

A browser with WebGPU and 16-bit floats on the GPU, which Chrome and Edge have, and the model's size of the browser's storage. A model lives only as long as its page, so each visit to local mode loads it again from that storage.

The engine's modules have been run under Node, through ONNX Runtime's WebGPU backend on an Apple Silicon Mac, where every interaction above was exercised on both models. It has not been tried in other browsers or on other GPUs.
