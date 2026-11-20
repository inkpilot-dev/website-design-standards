"""Inkport engine: ingestion, verification, training, gating and local inference.

The engine is a sidecar process. It speaks the NDJSON event protocol defined in
contracts/schemas/event.schema.json on stdout and is supervised by the Rust app.
"""

__version__ = "0.1.0"
