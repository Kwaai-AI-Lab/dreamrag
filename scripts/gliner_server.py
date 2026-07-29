#!/usr/bin/env python3
"""
gliner_server.py — HTTP server for GLiNER named entity recognition.

Exposes a simple REST API:
  POST /ner
  Request:  {"text": "...", "labels": ["person"], "threshold": 0.4}
  Response: [{"text": "Joe Rassool", "label": "person", "score": 0.92}, ...]

Usage:
  python3 scripts/gliner_server.py [--port 8000] [--model urchade/gliner_small-v2.1]
"""

import argparse
import json
import logging
from pathlib import Path

from flask import Flask, request, jsonify
import torch
from gliner import GLiNER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None


def load_model(model_name: str) -> GLiNER:
    """Load GLiNER model."""
    logger.info(f"Loading GLiNER model: {model_name}")
    try:
        m = GLiNER.from_pretrained(model_name)
        if torch.cuda.is_available():
            m.to("cuda")
            logger.info("Model loaded on CUDA")
        else:
            logger.info("Model loaded on CPU")
        return m
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": model.__class__.__name__ if model else "unloaded"}), 200


@app.route("/ner", methods=["POST"])
def ner():
    """Named entity recognition endpoint.

    Expected JSON:
    {
        "text": "John Smith works at OpenAI.",
        "labels": ["person", "organization"],
        "threshold": 0.5
    }

    Returns:
    [
        {"text": "John Smith", "label": "person", "score": 0.95},
        {"text": "OpenAI", "label": "organization", "score": 0.98}
    ]
    """
    if not model:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        text = data.get("text", "")
        labels = data.get("labels", [])
        threshold = float(data.get("threshold", 0.5))

        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400

        if not labels:
            return jsonify({"error": "Missing 'labels' field"}), 400

        # Run NER
        entities = model.predict_entities(text, labels, threshold=threshold)

        # Format response
        results = [
            {
                "text": ent["text"],
                "label": ent["label"],
                "score": float(ent["score"]),
            }
            for ent in entities
        ]

        return jsonify(results), 200

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON"}), 400
    except Exception as e:
        logger.exception("Error processing NER request")
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="GLiNER HTTP server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--model", default="urchade/gliner_small-v2.1", help="GLiNER model to load")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    global model
    model = load_model(args.model)

    logger.info(f"Starting GLiNER server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
