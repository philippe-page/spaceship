"""HTTP server for the local engine. It starts with no model: a page lists the models (GET /health) and picks one (POST /load),
which is downloaded the first time and loaded on the model thread. Every other request becomes a job on that thread, since
MLX arrays belong to the thread that evaluates them; jobs are generators, stepped in turn, so a readout can run between two
tokens of a continuation. Responses stream newline-delimited JSON. A client that disconnects cancels its continuation at the
next token; a readout is a single step and runs to its end. Only pages served from this machine may use it, so a site open
in the browser cannot run the model."""

import gc
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx

import analysis
import models
from generation import complete
from model import Engine
from steering import CATALOG, Vectors

PORT = int(os.environ.get("SPACESHIP_ENGINE_PORT", "8765"))
ALLOWED_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")
DONE = object()


class Job:
    def __init__(self, name: str, steps):
        self.name = name
        self.steps = steps
        self.output = queue.Queue()
        self.cancelled = threading.Event()


class Scheduler:
    def __init__(self):
        self.incoming = queue.Queue()
        self.engine = None
        self.vectors = None

    def submit(self, job: Job):
        self.incoming.put(job)
        return job

    def run(self):
        active = []
        while True:
            try:
                while True:
                    active.append(self.incoming.get(block=not active))
            except queue.Empty:
                pass
            for job in list(active):
                if job.cancelled.is_set():
                    job.steps.close()
                    active.remove(job)
                    log(f"{job.name} cancelled")
                    continue
                try:
                    item = next(job.steps)
                    if item is not None:
                        job.output.put(item)
                except StopIteration:
                    job.output.put(DONE)
                    active.remove(job)
                except Exception as error:
                    traceback.print_exc()
                    job.output.put({"error": str(error)})
                    job.output.put(DONE)
                    active.remove(job)


scheduler = Scheduler()


def load_steps(choice: dict, path: str):
    """Replaces the loaded model with `choice`, unless it is the one loaded, and finds its dimension vectors. Jobs already
    running finish on the old one."""
    if scheduler.engine is None or scheduler.engine.id != choice["id"]:
        yield {"status": "loading"}
        started = time.time()
        scheduler.engine = scheduler.vectors = None
        # An engine and its layers refer to each other, so the old model's memory is only freed by a collection.
        gc.collect()
        mx.clear_cache()
        engine = Engine(choice["id"], path)
        vectors = Vectors(engine)
        for dimension in CATALOG:
            vectors.directions_of(dimension)
        scheduler.engine, scheduler.vectors = engine, vectors
        log(f"loaded {choice['repo']} and {len(CATALOG)} dimension vectors in {time.time() - started:.1f}s")
    yield {"status": "ready", "model": choice["id"]}


def steps_of(path: str, request: dict):
    engine, vectors = scheduler.engine, scheduler.vectors
    if path == "/complete":
        r = request
        yield from complete(engine, vectors, r["prefix"], r.get("leads", []), r.get("controls", []), r.get("compare", False))
    elif path == "/attention":
        yield from analysis.focus_boxes(engine, request["text"])
    elif path == "/influence":
        yield from analysis.influence(engine, request["sentences"], request["focus"])
    elif path == "/relevance":
        yield from analysis.relevance(engine, request["story"], request["tip"], request["leads"])
    elif path == "/realization":
        r = request
        yield from analysis.realization(engine, vectors, r["context"], r["sentences"], r["asked"], r["control"])
    elif path == "/dimensions":
        yield from vectors.menu(request["text"], request.get("taken", []), request["size"])
    else:
        raise KeyError(path)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path != "/health":
            return self.send_error(404)
        listed = [
            {"id": m["id"], "name": m["name"], "size": m["size"], "downloaded": models.local_path(m) is not None} for m in models.MODELS
        ]
        engine = scheduler.engine
        body = json.dumps({"model": engine.id if engine else None, "models": listed}).encode()
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.is_allowed():
            return self.send_error(403)
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/load":
            return self.load(request)
        if scheduler.engine is None:
            return self.stream([{"error": "no model loaded"}])
        started = time.time()
        job = scheduler.submit(Job(self.path, steps_of(self.path, request)))
        try:
            count = self.stream(job_items(job))
            log(f"{self.path} {count} items in {(time.time() - started) * 1000:.0f}ms")
        except ConnectionError:
            job.cancelled.set()

    def load(self, request: dict):
        """Streams the download's progress, if the model is not on this machine yet, then its loading. A page that leaves
        mid-download leaves the download running, so the model is there the next time it is picked."""
        choice = models.model_of(request.get("model", ""))
        if choice is None:
            return self.stream([{"error": f"unknown model {request.get('model')}"}])
        try:
            self.stream(load_items(choice))
        except ConnectionError:
            pass

    def stream(self, items):
        """Sends `items` as newline-delimited JSON, each as soon as it is ready; returns how many were sent."""
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        count = 0
        for item in items:
            self.chunk(json.dumps(item) + "\n")
            count += 1
        self.chunk("")
        return count

    def chunk(self, text: str):
        data = text.encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def is_allowed(self):
        """Requests without an origin come from tools on this machine, not from a page."""
        origin = self.headers.get("Origin")
        return origin is None or ALLOWED_ORIGIN.match(origin) is not None

    def cors(self):
        if not self.is_allowed() or self.headers.get("Origin") is None:
            return
        self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        pass


def job_items(job: Job):
    while (item := job.output.get()) is not DONE:
        yield item


def load_items(choice: dict):
    path = models.local_path(choice)
    if path is None:
        progress, result = {"done": 0}, {}

        def fetch():
            try:
                result["path"] = models.download(choice, progress)
            except Exception as error:
                result["error"] = str(error)

        log(f"downloading {choice['repo']}")
        fetching = threading.Thread(target=fetch, daemon=True)
        fetching.start()
        while fetching.is_alive():
            yield {"status": "downloading", "done": progress["done"], "total": choice["size"]}
            fetching.join(0.5)
        if "error" in result:
            yield {"error": f"download failed: {result['error']}"}
            return
        path = result["path"]
    yield from job_items(scheduler.submit(Job("/load", load_steps(choice, path))))


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        """The page drops connections whenever it cancels a request, which is routine, not an error worth a traceback."""
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def log(message: str):
    print(f"[engine] {message}", flush=True)


if __name__ == "__main__":
    threading.Thread(target=scheduler.run, daemon=True).start()
    server = Server(("127.0.0.1", PORT), Handler)
    log(f"listening on http://localhost:{PORT}, running on {mx.default_device()}")
    server.serve_forever()
