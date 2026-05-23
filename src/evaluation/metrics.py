import json
import pandas as pd
from pathlib import Path

def save_metrics(metrics: dict, out_json: Path, out_html: Path | None = None):
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if out_html:
        html = "<html><head><title>Model Metrics</title></head><body>"
        html += "<h1>Model Metrics</h1><pre>" + json.dumps(metrics, indent=2) + "</pre></body></html>"
        out_html.write_text(html, encoding="utf-8")

def metrics_report(metrics: dict) -> str:
    lines = [f"<h2>{k}</h2><ul>" + ''.join(f"<li>{kk}: {vv}</li>" for kk, vv in v.items()) + "</ul>" for k, v in metrics.items()]
    return "<html><body>" + ''.join(lines) + "</body></html>"def evaluate(predictions, targets):
    return {'metric': 'Metrics', 'value': 0.0}
