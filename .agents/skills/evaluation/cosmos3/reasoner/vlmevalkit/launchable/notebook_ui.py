# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped, theme-aware presentation for the public benchmark notebook."""
from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
import subprocess

from IPython.display import FileLink, HTML, display

from benchmarks import CATALOG, validate_selection

# Kaizen UI 1.19 semantic tokens, adapted to native notebook HTML. Jupyter owns
# the shell and theme; do not load KUI's global reset, fonts, or React runtime.
STYLE = """<style>
.ev-notebook {
  --ev-text:var(--text-color-primary,var(--jp-content-font-color1,#1a1a1a));
  --ev-secondary:var(--text-color-secondary,var(--jp-content-font-color2,#555));
  --ev-surface:var(--background-color-surface-base,var(--jp-layout-color1,#fff));
  --ev-subtle:var(--background-color-background-subtle,var(--jp-layout-color2,#f6f6f6));
  --ev-border:var(--border-color-base,var(--jp-border-color2,#d6d6d6));
  --ev-brand:var(--border-color-brand,var(--color-brand,#76b900));
  --ev-warning:var(--border-color-feedback-warning,var(--jp-warn-color1,#b58100));
  color:var(--ev-text);background:var(--ev-surface);
  font-family:var(--font-sans,var(--jp-content-font-family,sans-serif));
  font-size:var(--text-14,14px);line-height:var(--leading-lh-150,1.5);
  max-width:1120px;width:100%;margin:var(--spacing-density-xl,16px) auto;
  padding:var(--spacing-density-xl,16px);
}
.ev-notebook * {box-sizing:border-box}
.ev-notebook h1,.ev-notebook h2,.ev-notebook h3 {color:var(--ev-text);font-family:inherit;font-weight:var(--font-weight-bold,700);line-height:var(--leading-lh-125,1.25);margin:0}
.ev-notebook h1 {font-size:var(--text-32,32px);max-width:28em}
.ev-notebook h2 {font-size:var(--text-24,24px)}
.ev-notebook h3 {font-size:var(--text-18,18px)}
.ev-notebook p {margin:0}
.ev-notebook .ev-kicker,.ev-notebook .ev-muted {color:var(--ev-secondary);font-size:var(--text-12,12px)}
.ev-notebook .ev-kicker {font-weight:var(--font-weight-semibold,500)}
.ev-notebook .ev-heading {display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:var(--spacing-density-lg,12px);margin:var(--spacing-density-md,8px) 0 var(--spacing-density-xl,16px)}
.ev-notebook .ev-badge {display:inline-block;padding:var(--spacing-density-xs,4px) var(--spacing-density-md,8px);border-radius:var(--radius-md,4px);background:var(--ev-subtle);color:var(--ev-text);font-size:var(--text-12,12px);font-weight:var(--font-weight-semibold,500)}
.ev-notebook .ev-badge-warning {background:var(--background-color-feedback-warning,color-mix(in srgb,var(--ev-warning) 12%,var(--ev-surface)))}
.ev-notebook .ev-note {padding:var(--spacing-density-xl,16px);border:1px solid var(--ev-border);border-radius:var(--radius-md,4px);background:var(--ev-subtle);margin:var(--spacing-density-xl,16px) 0}
.ev-notebook .ev-warning {border-inline-start:var(--spacing-density-xs,4px) solid var(--ev-warning);background:var(--background-color-feedback-warning,color-mix(in srgb,var(--ev-warning) 8%,var(--ev-surface)))}
.ev-notebook .ev-value {font-size:var(--text-48,48px);line-height:var(--leading-lh-125,1.25);font-weight:var(--font-weight-bold,700);font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.ev-notebook .ev-metric {display:grid;gap:var(--spacing-density-md,8px);margin:var(--spacing-density-2xl,24px) 0}
.ev-notebook .ev-result {border:1px solid var(--ev-border);border-radius:var(--radius-lg,8px);padding:var(--spacing-density-2xl,24px)}
.ev-notebook .ev-facts {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr));gap:var(--spacing-density-xl,16px);margin:var(--spacing-density-xl,16px) 0}
.ev-notebook .ev-facts div {min-width:0}
.ev-notebook dt {display:block;float:none;width:auto;color:var(--ev-secondary);font-size:var(--text-12,12px);font-weight:var(--font-weight-regular,400);margin-bottom:var(--spacing-density-xs,4px)}
.ev-notebook dd {display:block;float:none;width:auto;margin:0;font-weight:var(--font-weight-semibold,500);overflow-wrap:anywhere}
.ev-notebook details {border-top:1px solid var(--ev-border);padding:var(--spacing-density-xl,16px) 0}
.ev-notebook details p,.ev-notebook details ul {margin-top:var(--spacing-density-lg,12px)}
.ev-notebook summary {cursor:pointer;font-weight:var(--font-weight-semibold,500)}
.ev-notebook :where(summary,a):focus-visible {outline:2px solid var(--ev-text);outline-offset:var(--spacing-density-xs,4px)}
.ev-notebook a {color:var(--jp-content-link-color,var(--ev-text));text-decoration:underline;text-underline-offset:3px}
.ev-notebook code,.ev-notebook pre {font-family:var(--font-mono,var(--jp-code-font-family,monospace));font-size:var(--text-12,12px)}
.ev-notebook code {overflow-wrap:anywhere;white-space:normal;background:var(--ev-subtle);padding:var(--spacing-density-xs,4px);border-radius:var(--radius-sm,2px)}
.ev-notebook pre {white-space:pre-wrap;overflow-wrap:anywhere;background:var(--ev-subtle);padding:var(--spacing-density-xl,16px);border-radius:var(--radius-md,4px)}
.ev-notebook details li+li {margin-top:var(--spacing-density-md,8px)}
</style>"""


def _html(body):
    # Each output is self-contained when displayed or rerun independently.
    display(HTML(STYLE + '<section class="ev-notebook">' + body + '</section>'))


def show_plan(benchmark, scope, data, work, workers, score_only=False):
    validate_selection(benchmark, scope, score_only)
    item = CATALOG[benchmark]
    _html('<div class="ev-heading"><h2>Run configuration</h2>'
          f'<span class="ev-badge">{escape(scope.title())}</span></div>'
          f'<p>{escape(item["scope"] if scope == "full" else item["smoke"])}</p>'
          '<dl class="ev-facts"><div><dt>Execution</dt>'
          f'<dd>{"Score existing predictions" if score_only else "Inference and local scoring"}</dd></div>'
          f'<div><dt>API workers</dt><dd>{workers}</dd></div></dl>'
          '<aside class="ev-note ev-warning"><strong>Download budget</strong><p>'
          f'{"Pinned metadata only; no videos or endpoint needed." if score_only else escape(item["download"])}</p></aside>'
          '<details open><summary>Protocol and files</summary>'
          f'<p>{escape(item["protocol"])}</p><dl class="ev-facts">'
          f'<div><dt>Dataset directory</dt><dd><code>{escape(str(data))}</code></dd></div>'
          f'<div><dt>New results directory</dt><dd><code>{escape(str(work))}</code></dd></div></dl></details>'
          '<details><summary>Source and attribution</summary>'
          f'<p>{escape(item["license"])}</p><a href="{escape(item["source"], quote=True)}">Publisher source</a>'
          '<p>Public hosting does not clear all underlying redistribution rights. '
          'Downloads stay on your instance; no raw data is bundled or uploaded by this notebook.</p></details>')
    if scope == "full":
        show_baseline(benchmark)


def show_baseline(benchmark):
    """Publish the pinned full-run target without presenting it as paper parity."""
    record = json.loads(Path(__file__).with_name("baselines.json").read_text())
    baseline = record["benchmarks"][benchmark]
    suffix = "%" if baseline["unit"] == "percent" else " correlation points"
    paper = baseline["paper_value"]
    comparison = ("No reference in the cited Cosmos3 table." if paper is None
                  else f'Paper reference: {paper:.2f}{suffix}.')
    _html('<aside class="ev-note"><h2>Expected full-run result</h2>'
          f'<p><strong>{baseline["value"]:.2f}{suffix}</strong> on '
          f'{baseline["examples"]:,} examples · {escape(record["model"])} · '
          f'{escape(record["model_revision"][:12])}</p>'
          f'<p class="ev-muted">Measured {escape(baseline.get("measurement_date", record["measurement_date"]))}. '
          f'{escape(comparison)}</p><p>{escape(baseline["paper_note"])}</p>'
          '<details><summary>Pinned baseline protocol</summary>'
          f'<p>{escape(baseline["protocol"])}</p>'
          f'<p>{escape(baseline["verification"])}</p></details></aside>')


def invoke(command, log_path=None):
    """Stream real subprocess progress; preserve failures and a private log."""
    log = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("x")
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    if log is not None:
                        log.write(line)
                        log.flush()
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
    finally:
        if log is not None:
            log.close()


def show_results(summary, benchmark, scope, work, log_path=None):
    """Bind the displayed public benchmark to its artifact."""
    validate_selection(benchmark, scope)
    _show_results(summary, benchmark, scope, work, CATALOG[benchmark], log_path)


def show_customer_results(summary, work, log_path=None):
    """Keep customer results separate from official benchmark measurements."""
    item = {"title": "Customer-defined video evaluation",
            "metric": "Multiple-choice accuracy"}
    _show_results(summary, "customer-video-mcq", "custom", work, item, log_path)


def _execution_facts(summary):
    evidence = summary.get("execution_evidence")
    if not isinstance(evidence, dict):
        return ""
    requests = [entry["settings"] for entry in evidence.get("requested_settings", [])
                if isinstance(entry.get("settings"), dict)]
    sampling = summary.get("sampling", {})

    def formatted(values, missing="Not observed"):
        values = sorted({value for value in values if isinstance(value, (int, float))
                         and not isinstance(value, bool) and math.isfinite(value)})
        if not values:
            return missing
        return f"{values[0]:,}" if len(values) == 1 else f"{values[0]:,}–{values[-1]:,}"

    requested_frames = [sampling.get("nframes")] + [
        item.get("media", {}).get("expected_frame_slots", item.get("media", {}).get("sampled_frames"))
        or item.get("image_inputs") or None
        for item in requests]
    minima = {sampling.get("min_pixels")} | {item.get("mm_processor_kwargs", {}).get("min_pixels") for item in requests}
    maxima = {sampling.get("max_pixels")} | {item.get("mm_processor_kwargs", {}).get("max_pixels") for item in requests}
    minima.discard(None)
    maxima.discard(None)
    budget = (formatted(minima | maxima, "Not recorded") if minima and maxima else
              "≤" + formatted(maxima) if maxima else "≥" + formatted(minima) if minima else "Not recorded")
    shapes = evidence.get("observed_visual_shapes", [])
    areas = [area for shape in shapes for key in ("image_grid_pixel_areas", "video_temporal_grid_pixel_areas")
             for area in shape.get(key, [])]
    facts = {
        "Requested input frames/images": formatted(requested_frames, "Not recorded"),
        "Observed image groups / video slots": formatted(
            shape.get("video_frame_slots") or len(shape.get("image_group_tokens", [])) or None for shape in shapes),
        "Requested spatial budget (pixels/frame)": budget,
        "Observed spatial grid area (pixels)": formatted(areas),
        "Returned model": ", ".join(evidence.get("observed_models", [])) or "Not observed",
        "Response evidence": f'{evidence.get("requests_recorded", 0)} / {evidence.get("expected_requests", "?")} requests',
    }
    return ('<h3>Input execution evidence</h3><dl class="ev-facts">' +
            "".join(f"<div><dt>{escape(key)}</dt><dd>{escape(str(value))}</dd></div>" for key, value in facts.items()) +
            '</dl><p class="ev-muted">Grid areas and frame slots are conditional on the explicit processor profile. '
            'They do not attest decoded/unique frames, width/height, or generation settings. '
            'See the detailed record for unknowns and geometry differences.</p>')


def _show_results(summary, benchmark, scope, work, item, log_path=None):
    if summary.get("benchmark") != benchmark or summary.get("scope") != scope:
        raise ValueError("Result identity does not match this notebook's selected benchmark/scope")
    if summary.get("status") not in {"completed", "completed_with_generation_issues"}:
        raise ValueError("This run is not complete; inspect retained diagnostics before reporting a score")
    metric = summary["primary_metric"]
    value, unit = metric["value"], metric["unit"]
    if value is not None and not math.isfinite(value):
        raise ValueError("Metric must be finite or explicitly undefined")
    headline = "Not defined" if value is None else f"{value:.2f}{'%' if unit == 'percent' else ''}"
    coverage = summary["coverage"]
    if coverage["scored"] != coverage["expected"]:
        raise ValueError("Refusing to present incomplete coverage as a completed benchmark")
    warnings = list(summary.get("warnings", []))
    issues = summary["status"] == "completed_with_generation_issues"
    notes = "".join(f'<li>{escape(str(note))}</li>' for note in warnings)
    detail = "Correlation ×100 — not accuracy %" if unit == "pearson_x100" else "Accuracy across all selected questions"
    if value is None:
        detail = "Insufficient rating variance for correlation; inspect the retained predictions."
    elapsed = summary.get("elapsed_seconds")
    elapsed_text = "Not recorded" if elapsed is None else f"{elapsed / 60:.1f} min"
    status = "Completed with generation issues" if issues else "Run completed"
    banner = ("Generation issues were retained in the score; inspect the notes below. " if issues else "")
    if scope == "smoke":
        banner += "Smoke run only: this checks the pipeline, not full-benchmark performance."
    if scope == "custom":
        banner += "Customer-defined labeled data: not an official benchmark or leaderboard result."
    execution = {
        "requested": summary.get("requested_execution"),
        "effective": summary.get("effective_execution"),
        "evidence": summary.get("execution_evidence"),
    }
    notice = (f'<aside class="ev-note ev-warning">{escape(banner)}</aside>' if banner else "")
    _html(f'<div class="ev-result" data-benchmark="{benchmark}" data-scope="{scope}">'
          '<p class="ev-kicker">Benchmark result</p>'
          f'<div class="ev-heading"><h2>{escape(item["title"])}</h2>'
          f'<span class="ev-badge{" ev-badge-warning" if issues else ""}">{status}</span></div>'
          f'<p class="ev-muted">{escape(scope.title())} scope · {escape(benchmark)}</p>'
          f'{notice}<div class="ev-metric"><strong>{escape(item["metric"])}</strong>'
          f'<div class="ev-value">{escape(headline)}</div>'
          f'<p class="ev-muted">{escape(detail)}</p></div>'
          '<dl class="ev-facts"><div><dt>Selected coverage</dt>'
          f'<dd>{coverage["scored"]:,} of {coverage["expected"]:,} examples</dd></div>'
          f'<div><dt>Measured run time</dt><dd>{escape(elapsed_text)}</dd></div></dl>'
          f'{_execution_facts(summary)}'
          '<p class="ev-muted">Published-score equivalence has not been established.</p>'
          f'<details{" open" if issues else ""}><summary>Data and scoring notes</summary><ul>{notes}</ul></details>'
          '<details><summary>Exact protocol and metric record</summary>'
          f'<pre>{escape(json.dumps({"primary_metric": metric, "protocol": summary["protocol"]}, indent=2))}</pre>'
          '</details><details><summary>Requested and effective execution settings</summary>'
          f'<pre>{escape(json.dumps(execution, indent=2))}</pre>'
          '<p class="ev-muted">Null means not recorded in this artifact, not verified. '
          'Effective input geometry is derived from observed processor tokens when supported; '
          'requested generation parameters are not a server attestation.</p></details></div>')
    # These links are private runtime artifacts, never source files to publish.
    artifacts = [Path(work) / "notebook-summary.json", Path(work) / "summary.json",
                 Path(work) / "provenance.json", Path(work) / "run.json",
                 Path(work) / "scores" / "summary.json", Path(work) / "run.log",
                 Path(work) / "execution-summary.json",
                 Path(work) / "requested-execution.json", Path(work) / "effective-execution.json"]
    if log_path is not None:
        artifacts.append(Path(log_path))
    for artifact in artifacts:
        if artifact.is_file():
            display(FileLink(str(artifact), url_prefix="/files"))
