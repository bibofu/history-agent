from __future__ import annotations

import json
import platform
import sys
from typing import Annotated, Literal

import typer

from history_agent import __version__
from history_agent.answering.service import check_deepseek_connection
from history_agent.config import get_settings
from history_agent.corpus.catalog import load_catalog
from history_agent.corpus.exporter import export_diff, export_manifest, load_latest_diff
from history_agent.corpus.scanner import scan_corpus
from history_agent.db import Database
from history_agent.evaluation.answers import evaluate_answers
from history_agent.evaluation.retrieval import evaluate_retrieval
from history_agent.extraction.benchmark import benchmark_parsers
from history_agent.extraction.full import extract_all_text
from history_agent.extraction.ocr import extract_all_ocr
from history_agent.extraction.report import build_quality_report
from history_agent.extraction.sample import extract_sample_pages
from history_agent.log import configure_logging
from history_agent.processing.chunks import build_all_chunks
from history_agent.research.catalog import load_person_catalog, load_relation_type_catalog
from history_agent.research.chronology import extract_chronology_events
from history_agent.research.event_deduplication import (
    list_event_merge_review_queue,
    merge_duplicate_events,
    review_event_merge,
)
from history_agent.research.model_extraction import (
    enrich_events_with_model,
    list_review_queue,
)
from history_agent.research.people import (
    propose_person_merge,
    resolve_person,
    review_person_merge,
    sync_person_catalog,
    sync_relation_types,
)
from history_agent.retrieval.hybrid import search_hybrid_index
from history_agent.retrieval.keyword import build_keyword_index, search_keyword_index
from history_agent.retrieval.vector import build_vector_index, search_vector_index
from history_agent.run_tracker import RunTracker

app = typer.Typer(
    name="history-agent",
    help="Evidence-grounded local research tools for the 1921-1978 corpus.",
    no_args_is_help=True,
)
corpus_app = typer.Typer(help="Inspect and register local PDF sources.", no_args_is_help=True)
extract_app = typer.Typer(help="Extract page-level text and route OCR work.", no_args_is_help=True)
process_app = typer.Typer(help="Clean, structure, and chunk extracted text.", no_args_is_help=True)
index_app = typer.Typer(help="Build local retrieval indexes.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate retrieval and grounded answers.", no_args_is_help=True)
llm_app = typer.Typer(help="Inspect and test the DeepSeek V4 connection.", no_args_is_help=True)
research_app = typer.Typer(
    help="Manage auditable people, events, and relationships.", no_args_is_help=True
)
people_app = typer.Typer(help="Manage stable people and aliases.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(extract_app, name="extract")
app.add_typer(process_app, name="process")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
app.add_typer(llm_app, name="llm")
app.add_typer(research_app, name="research")
research_app.add_typer(people_app, name="people")


@app.callback()
def configure_console() -> None:
    """Use UTF-8 for Chinese filenames on Windows terminals."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


@app.command()
def health(json_output: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Check paths and show the active, non-secret configuration."""

    settings = get_settings()
    configure_logging(settings.log_level, json_output=json_output)
    payload = {
        "status": (
            "ok" if settings.docs_dir.is_dir() and settings.catalog_path.is_file() else "degraded"
        ),
        "version": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "docs_exists": settings.docs_dir.is_dir(),
        "catalog_exists": settings.catalog_path.is_file(),
        "settings": settings.public_snapshot(),
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, value in payload.items():
            typer.echo(f"{key}: {value}")


@corpus_app.command("scan")
def corpus_scan(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
    accept_changes: bool = typer.Option(
        False,
        "--accept-changes",
        help="Register PDFs whose content hash changed after reviewing the diff.",
    ),
) -> None:
    """Scan docs/, update SQLite metadata, and write manifest/diff reports."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(settings.runs_dir, "corpus_scan", settings.public_snapshot())
    try:
        database = Database(settings.database_path)
        catalog = load_catalog(settings.catalog_path)
        summary = scan_corpus(
            database=database,
            catalog=catalog,
            docs_dir=settings.docs_dir,
            project_root=settings.project_root,
            run_id=tracker.run_id,
            accept_changes=accept_changes,
        )
        manifest_json, manifest_csv = export_manifest(database, settings.reports_dir)
        diff_run, diff_latest = export_diff(summary, settings.reports_dir)
        payload = {
            "run_id": summary.run_id,
            "counts": summary.counts,
            "manifest_json": str(manifest_json),
            "manifest_csv": str(manifest_csv),
            "diff_report": str(diff_run),
            "latest_diff": str(diff_latest),
            "observations": [item.model_dump() for item in summary.observations],
        }
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"scan run: {summary.run_id}")
        typer.echo(f"counts: {summary.counts}")
        typer.echo(f"manifest: {manifest_json}")
        typer.echo(f"diff: {diff_run}")


@corpus_app.command("export")
def corpus_export() -> None:
    """Export the current SQLite corpus manifest to JSON and CSV."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    database = Database(settings.database_path)
    database.initialize()
    json_path, csv_path = export_manifest(database, settings.reports_dir)
    typer.echo(str(json_path))
    typer.echo(str(csv_path))


@corpus_app.command("diff")
def corpus_diff(json_output: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Show the latest completed corpus scan."""

    settings = get_settings()
    database = Database(settings.database_path)
    database.initialize()
    payload = load_latest_diff(database)
    if payload is None:
        typer.echo("No corpus scan has been recorded.")
        raise typer.Exit(code=1)
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"run_id: {payload['run_id']}")
        typer.echo(f"counts: {payload['counts']}")
        for item in payload["observations"]:
            typer.echo(f"{item['status']:>12}  {item['filename']}")


@research_app.command("init")
def research_initialize(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Create research tables and sync people and relation vocabularies."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    database = Database(settings.database_path)
    people = load_person_catalog(settings.person_aliases_path)
    relations = load_relation_type_catalog(settings.relation_types_path)
    payload = {
        **sync_person_catalog(database, people),
        **sync_relation_types(database, relations),
        "database": str(settings.database_path),
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(
            f"people: {payload['people']}; aliases: {payload['aliases']}; "
            f"ambiguities: {payload['ambiguities']}; "
            f"relation types: {payload['relation_types']}"
        )


@research_app.command("extract-chronologies")
def research_extract_chronologies(
    document_id: Annotated[
        list[str] | None,
        typer.Option(
            "--document-id",
            help="Extract one supported chronology. Repeat for multiple documents.",
        ),
    ] = None,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Extract auditable event candidates from supported chronology layouts."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(
        settings.runs_dir,
        "extract_chronology_events",
        settings.public_snapshot(),
    )
    try:
        database = Database(settings.database_path)
        sync_person_catalog(
            database, load_person_catalog(settings.person_aliases_path)
        )
        summary = extract_chronology_events(
            database=database,
            pages_dir=settings.pages_dir,
            ocr_dir=settings.ocr_dir,
            structure_dir=settings.structure_dir,
            events_dir=settings.events_dir,
            reports_dir=settings.reports_dir,
            person_aliases_path=settings.person_aliases_path,
            run_id=tracker.run_id,
            research_start=settings.research_start.year,
            research_end=settings.research_end.year,
            document_ids=document_id,
        )
        payload = summary.model_dump()
        payload["totals"] = summary.totals
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(f"chronology extraction run: {summary.run_id}")
    typer.echo(f"totals: {summary.totals}")
    for item in summary.documents:
        typer.echo(
            f"{item.document_id}: candidates={item.candidates}, "
            f"created={item.database_created}, updated={item.database_updated}, "
            f"skipped={item.database_skipped}, "
            f"locations={item.location_candidates}"
        )
    typer.echo(str(settings.reports_dir / "chronology_extraction_latest.json"))


@research_app.command("enrich-events")
def research_enrich_events(
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            max=50,
            help="Process at most this many candidates; deliberately bounded to control cost.",
        ),
    ] = 5,
    document_id: Annotated[
        list[str] | None,
        typer.Option(
            "--document-id",
            help="Only process events from this source. Repeat for multiple documents.",
        ),
    ] = None,
    event_id: Annotated[
        list[str] | None,
        typer.Option(
            "--event-id",
            help="Only process this event ID. Repeat for multiple events.",
        ),
    ] = None,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed",
            help="Retry prior invalid or failed attempts; successful attempts still skip.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show selected candidates without calling DeepSeek or changing research rows.",
        ),
    ] = False,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Use grounded DeepSeek JSON output to enrich rule-generated events."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(
        settings.runs_dir,
        "model_event_extraction",
        {
            **settings.public_snapshot(),
            "limit": limit,
            "document_ids": document_id or [],
            "event_ids": event_id or [],
            "retry_failed": retry_failed,
            "dry_run": dry_run,
        },
    )
    try:
        database = Database(settings.database_path)
        catalog = load_person_catalog(settings.person_aliases_path)
        sync_person_catalog(database, catalog)
        summary = enrich_events_with_model(
            database=database,
            settings=settings,
            catalog=catalog,
            reports_dir=settings.reports_dir,
            run_id=tracker.run_id,
            limit=limit,
            document_ids=document_id,
            event_ids=event_id,
            retry_failed=retry_failed,
            dry_run=dry_run,
        )
        payload = summary.model_dump()
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(f"model extraction run: {summary.run_id}")
    typer.echo(
        f"selected={summary.selected}, succeeded={summary.succeeded}, "
        f"invalid={summary.invalid}, failed={summary.failed}, "
        f"skipped_prior={summary.skipped_prior_attempt}"
    )
    typer.echo(f"tokens: {summary.total_tokens}")
    for selected_event_id in summary.event_ids:
        typer.echo(selected_event_id)
    typer.echo(str(settings.reports_dir / "model_event_extraction_latest.json"))


@research_app.command("review-queue")
def research_review_queue(
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 20,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List pending model-assisted event review items."""

    settings = get_settings()
    rows = list_review_queue(Database(settings.database_path), limit=limit)
    if json_output:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        typer.echo("No pending event review items.")
        return
    for row in rows:
        typer.echo(
            f"P{row['priority']} {row['event_id']} {row['start_value'] or '-'} "
            f"{row['name']} [{', '.join(row['reason_codes'])}]"
        )


@research_app.command("merge-events")
def research_merge_events(
    minimum_score: Annotated[
        float,
        typer.Option(
            "--minimum-score",
            min=0.5,
            max=0.95,
            help="Minimum score for exact-day duplicate candidates.",
        ),
    ] = 0.70,
    automatic_score: Annotated[
        float,
        typer.Option(
            "--automatic-score",
            min=0.5,
            max=0.99,
            help="Minimum score for a conservative high-confidence merge.",
        ),
    ] = 0.80,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Calculate candidates and report them without changing canonical tables.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Build non-destructive canonical events from cross-source duplicates."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(
        settings.runs_dir,
        "merge_duplicate_events",
        {
            **settings.public_snapshot(),
            "minimum_score": minimum_score,
            "automatic_score": automatic_score,
            "dry_run": dry_run,
        },
    )
    try:
        summary = merge_duplicate_events(
            database=Database(settings.database_path),
            reports_dir=settings.reports_dir,
            run_id=tracker.run_id,
            minimum_score=minimum_score,
            automatic_score=automatic_score,
            dry_run=dry_run,
        )
        payload = summary.model_dump()
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(f"event merge run: {summary.run_id}")
    typer.echo(
        f"pairs={summary.candidate_pairs}, groups={summary.candidate_groups}, "
        f"high_confidence={summary.high_confidence_groups}, "
        f"uncertain={summary.uncertain_groups}"
    )
    typer.echo(
        f"created={summary.created}, updated={summary.updated}, "
        f"skipped={summary.skipped}, protected={summary.protected}, "
        f"deactivated={summary.deactivated}"
    )
    typer.echo(str(settings.reports_dir / "event_merge_latest.json"))


@research_app.command("merge-review-queue")
def research_merge_review_queue(
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 20,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List uncertain canonical-event merges awaiting human review."""

    settings = get_settings()
    rows = list_event_merge_review_queue(
        Database(settings.database_path), limit=limit
    )
    if json_output:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        typer.echo("No pending event-merge review items.")
        return
    for row in rows:
        typer.echo(
            f"P{row['priority']} {row['canonical_event_id']} "
            f"score={row['merge_confidence']:.3f} members={row['member_count']} "
            f"{row['start_value'] or '-'} {row['name']}"
        )


@research_app.command("review-event-merge")
def research_review_event_merge(
    canonical_event_id: Annotated[str, typer.Argument()],
    decision: Annotated[
        Literal["confirmed", "rejected", "reopened"],
        typer.Option("--decision"),
    ],
    reviewed_by: Annotated[str, typer.Option("--reviewed-by")],
    note: Annotated[str | None, typer.Option("--note")] = None,
) -> None:
    """Confirm, reject, or reopen a canonical merge without changing source events."""

    settings = get_settings()
    review_id = review_event_merge(
        Database(settings.database_path),
        canonical_event_id=canonical_event_id,
        decision=decision,
        reviewed_by=reviewed_by,
        note=note,
    )
    typer.echo(f"{review_id}: {canonical_event_id} -> {decision}")


@people_app.command("resolve")
def people_resolve(
    name: Annotated[str, typer.Argument(help="Canonical name or historical alias.")],
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Resolve a name without forcing ambiguous aliases into one person."""

    settings = get_settings()
    resolution = resolve_person(Database(settings.database_path), name)
    if json_output:
        typer.echo(resolution.model_dump_json(indent=2))
        return
    typer.echo(f"status: {resolution.status}")
    for candidate in resolution.candidates:
        suffix = (
            f" -> {candidate.merged_into_person_id}"
            if candidate.merged_into_person_id
            else ""
        )
        typer.echo(
            f"{candidate.person_id}: {candidate.canonical_name} "
            f"(matched: {candidate.matched_form}; {candidate.status}{suffix})"
        )


@people_app.command("propose-merge")
def people_propose_merge(
    source_person_id: Annotated[str, typer.Argument()],
    target_person_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason", help="Why these records may match.")],
    proposed_by: Annotated[str, typer.Option("--proposed-by")] = "local_researcher",
) -> None:
    """Record a merge proposal without changing either person."""

    settings = get_settings()
    proposal_id = propose_person_merge(
        Database(settings.database_path),
        source_person_id=source_person_id,
        target_person_id=target_person_id,
        reason=reason,
        proposed_by=proposed_by,
    )
    typer.echo(proposal_id)


@people_app.command("review-merge")
def people_review_merge(
    proposal_id: Annotated[str, typer.Argument()],
    decision: Annotated[Literal["accepted", "rejected"], typer.Option("--decision")],
    reviewed_by: Annotated[str, typer.Option("--reviewed-by")],
    note: Annotated[str | None, typer.Option("--note")] = None,
) -> None:
    """Accept or reject a pending merge while preserving its audit record."""

    settings = get_settings()
    review_person_merge(
        Database(settings.database_path),
        proposal_id=proposal_id,
        decision=decision,
        reviewed_by=reviewed_by,
        review_note=note,
    )
    typer.echo(f"{proposal_id}: {decision}")


@extract_app.command("sample")
def extract_sample(json_output: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Extract representative text, mixed, and scan pages without running OCR."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(settings.runs_dir, "extract_sample", settings.public_snapshot())
    try:
        database = Database(settings.database_path)
        database.initialize()
        summary = extract_sample_pages(
            database=database,
            project_root=settings.project_root,
            output_dir=settings.samples_dir,
            reports_dir=settings.reports_dir,
            run_id=tracker.run_id,
        )
        payload = summary.model_dump()
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"sample run: {summary.run_id}")
        for item in summary.documents:
            typer.echo(f"{item.document_id}: {item.status_counts} -> {item.output_path}")


@extract_app.command("text")
def extract_text(
    document_id: Annotated[
        list[str] | None,
        typer.Option(
            "--document-id",
            help="Extract only this document ID. Repeat the option for multiple documents.",
        ),
    ] = None,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help="Ignore reusable page records and rebuild selected documents.",
        ),
    ] = False,
    parser_name: Annotated[
        Literal["pymupdf", "pypdf"],
        typer.Option("--parser", help="Primary text parser."),
    ] = "pymupdf",
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Extract all PDF pages to resumable, page-aligned JSONL files."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(settings.runs_dir, "extract_text", settings.public_snapshot())
    try:
        database = Database(settings.database_path)
        database.initialize()
        summary = extract_all_text(
            database=database,
            project_root=settings.project_root,
            output_dir=settings.pages_dir,
            reports_dir=settings.reports_dir,
            run_id=tracker.run_id,
            document_ids=document_id,
            rebuild=rebuild,
            parser_name=parser_name,
        )
        payload = summary.model_dump()
        payload["totals"] = summary.totals
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"text extraction run: {summary.run_id}")
        typer.echo(f"totals: {summary.totals}")
        for item in summary.documents:
            typer.echo(
                f"{item.document_id}: {item.status_counts}, "
                f"processed={item.processed_pages}, reused={item.reused_pages}"
            )


@extract_app.command("benchmark")
def extract_benchmark(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Compare pypdf, PyMuPDF, and pdfplumber on representative pages."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(settings.runs_dir, "parser_benchmark", settings.public_snapshot())
    try:
        database = Database(settings.database_path)
        database.initialize()
        payload = benchmark_parsers(
            database=database,
            project_root=settings.project_root,
            reports_dir=settings.reports_dir,
            run_id=tracker.run_id,
        )
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"parser benchmark run: {payload['run_id']}")
        for page in payload["pages"]:
            counts = {
                parser: result.get("character_count") for parser, result in page["parsers"].items()
            }
            typer.echo(f"{page['document_id']} p{page['pdf_page']}: {counts}")


@extract_app.command("report")
def extract_report(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Summarize text extraction quality and the current OCR work queue."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(settings.runs_dir, "extraction_quality_report", settings.public_snapshot())
    try:
        database = Database(settings.database_path)
        database.initialize()
        payload = build_quality_report(
            database=database,
            pages_dir=settings.pages_dir,
            ocr_dir=settings.ocr_dir,
            reports_dir=settings.reports_dir,
            run_id=tracker.run_id,
        )
        tracker.finish(payload["totals"])
    except Exception as exc:
        tracker.fail(exc)
        raise

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"quality report run: {payload['run_id']}")
        typer.echo(f"totals: {payload['totals']}")
        typer.echo(str(settings.reports_dir / "extraction_quality_latest.md"))


@extract_app.command("ocr")
def extract_ocr(
    document_id: Annotated[
        list[str] | None,
        typer.Option(
            "--document-id",
            help="OCR only this document ID. Repeat the option for multiple documents.",
        ),
    ] = None,
    pdf_page: Annotated[
        list[int] | None,
        typer.Option("--page", min=1, help="OCR a physical PDF page; requires one document."),
    ] = None,
    max_pages: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Process at most this many new candidate pages."),
    ] = None,
    dpi: Annotated[
        int,
        typer.Option("--dpi", min=72, max=300, help="PDF render resolution for OCR."),
    ] = 110,
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Ignore reusable OCR records for selected pages."),
    ] = False,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Run resumable PaddleOCR on pages routed from image-only PDFs."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(settings.runs_dir, "extract_ocr", settings.public_snapshot())
    try:
        database = Database(settings.database_path)
        database.initialize()
        summary = extract_all_ocr(
            database=database,
            project_root=settings.project_root,
            pages_dir=settings.pages_dir,
            output_dir=settings.ocr_dir,
            reports_dir=settings.reports_dir,
            run_id=tracker.run_id,
            document_ids=document_id,
            selected_pages=set(pdf_page) if pdf_page else None,
            max_pages=max_pages,
            dpi=dpi,
            rebuild=rebuild,
        )
        payload = summary.model_dump()
        payload["totals"] = summary.totals
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"OCR run: {summary.run_id}")
        typer.echo(f"totals: {summary.totals}")
        for item in summary.documents:
            typer.echo(
                f"{item.document_id}: {item.status_counts}, processed={item.processed_pages}, "
                f"reused={item.reused_pages}, remaining={item.remaining_pages}"
            )


@process_app.command("chunks")
def process_chunks(
    document_id: Annotated[
        list[str] | None,
        typer.Option(
            "--document-id",
            help="Build only this document ID. Repeat for multiple documents.",
        ),
    ] = None,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Recover PDF structure and build page-aligned retrieval chunks."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(settings.runs_dir, "build_chunks", settings.public_snapshot())
    try:
        database = Database(settings.database_path)
        database.initialize()
        summary = build_all_chunks(
            database=database,
            project_root=settings.project_root,
            pages_dir=settings.pages_dir,
            ocr_dir=settings.ocr_dir,
            chunks_dir=settings.chunks_dir,
            structure_dir=settings.structure_dir,
            reports_dir=settings.reports_dir,
            aliases_path=settings.person_aliases_path,
            run_id=tracker.run_id,
            research_start=settings.research_start.year,
            research_end=settings.research_end.year,
            document_ids=document_id,
        )
        payload = summary.model_dump()
        payload["totals"] = summary.totals
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"chunk build run: {summary.run_id}")
        typer.echo(f"totals: {summary.totals}")
        for item in summary.documents:
            typer.echo(
                f"{item.document_id}: pages={item.effective_pages}, "
                f"chunks={item.chunks}, structure={item.structure_entries}"
            )


@index_app.command("build-keyword")
def index_build_keyword(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Build an atomic Chinese bigram FTS5/BM25 index from all chunks."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(settings.runs_dir, "build_keyword_index", settings.public_snapshot())
    try:
        summary = build_keyword_index(
            chunks_dir=settings.chunks_dir,
            index_path=settings.keyword_index_path,
            reports_dir=settings.reports_dir,
            run_id=tracker.run_id,
        )
        payload = summary.model_dump()
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"keyword index run: {summary.run_id}")
        typer.echo(f"chunks: {summary.chunks}, documents: {summary.documents}")
        typer.echo(f"index: {summary.index_path} ({summary.size_bytes} bytes)")


@index_app.command("build-vector")
def index_build_vector(
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, max=512, help="Embedding batch size."),
    ] = 64,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Build an atomic local Qdrant index with a Chinese embedding model."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, json_output=json_output)
    tracker = RunTracker(settings.runs_dir, "build_vector_index", settings.public_snapshot())
    try:
        summary = build_vector_index(
            chunks_dir=settings.chunks_dir,
            index_path=settings.vector_index_path,
            model_cache_dir=settings.model_cache_dir / "fastembed",
            reports_dir=settings.reports_dir,
            run_id=tracker.run_id,
            batch_size=batch_size,
        )
        payload = summary.model_dump()
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"vector index run: {summary.run_id}")
        typer.echo(f"model: {summary.model_name}; chunks: {summary.chunks}")
        typer.echo(f"index: {summary.index_path} ({summary.size_bytes} bytes)")


@eval_app.command("retrieval")
def eval_retrieval(
    question_set: Annotated[
        str,
        typer.Option("--question-set", help="Question-set path relative to project root."),
    ] = "evals/retrieval_questions.json",
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 10,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Measure hybrid retrieval recall and target-document rank."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    path = settings.project_root / question_set
    tracker = RunTracker(settings.runs_dir, "evaluate_retrieval", settings.public_snapshot())
    try:
        payload = evaluate_retrieval(
            settings=settings,
            question_set_path=path,
            top_k=top_k,
            run_id=tracker.run_id,
        )
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        recall = payload["recall_at_k"]
        mrr = payload["mean_reciprocal_rank"]
        assert isinstance(recall, (int, float))
        assert isinstance(mrr, (int, float))
        typer.echo(f"retrieval evaluation run: {payload['run_id']}")
        typer.echo(
            f"questions: {payload['questions']}; "
            f"answerable: {payload['evaluated_questions']}; hits: {payload['hits']}"
        )
        typer.echo(f"Recall@{top_k}: {recall:.2%}")
        typer.echo(f"MRR: {mrr:.4f}")
        typer.echo(str(settings.reports_dir / "retrieval_eval_latest.json"))


@eval_app.command("answers")
def eval_answers(
    question_set: Annotated[
        str,
        typer.Option("--question-set", help="Question-set path relative to project root."),
    ] = "evals/retrieval_questions.json",
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=12)] = 10,
    with_llm: bool = typer.Option(
        False,
        "--with-llm",
        help="Evaluate DeepSeek output; default uses deterministic extractive answers.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Evaluate answerability, page citations, grounding, and required facts."""

    settings = get_settings()
    settings.ensure_runtime_dirs()
    path = settings.project_root / question_set
    tracker = RunTracker(settings.runs_dir, "evaluate_answers", settings.public_snapshot())
    try:
        payload = evaluate_answers(
            settings=settings,
            question_set_path=path,
            top_k=top_k,
            run_id=tracker.run_id,
            use_llm=with_llm,
        )
        tracker.finish(payload)
    except Exception as exc:
        tracker.fail(exc)
        raise
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    typer.echo(f"answer evaluation run: {payload['run_id']}")
    typer.echo(f"questions: {payload['questions']}; passed: {payload['passed']}")
    typer.echo(f"gold page hit rate: {float(metrics['gold_page_hit_rate']):.2%}")
    typer.echo(
        f"citation page accuracy: {float(metrics['citation_page_accuracy']):.2%}"
    )
    typer.echo(f"refusal accuracy: {float(metrics['refusal_accuracy']):.2%}")
    typer.echo(str(settings.reports_dir / "mvp_eval_latest.md"))


@llm_app.command("status")
def llm_status(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show non-secret DeepSeek configuration without making a network call."""

    settings = get_settings()
    payload = {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "thinking": settings.llm_thinking,
        "reasoning_effort": settings.llm_reasoning_effort,
        "max_tokens": settings.llm_max_tokens,
        "configured": settings.llm_enabled,
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@llm_app.command("check")
def llm_check(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Send a minimal request to verify the configured DeepSeek API key."""

    payload = check_deepseek_connection(get_settings())
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, value in payload.items():
            typer.echo(f"{key}: {value}")
    if payload["status"] != "ok":
        raise typer.Exit(code=1)


@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Chinese historical research query.")],
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 10,
    document_id: Annotated[
        list[str] | None,
        typer.Option("--document-id", help="Restrict to one or more document IDs."),
    ] = None,
    include_out_of_scope: Annotated[
        bool,
        typer.Option("--include-out-of-scope", help="Include evidence outside 1921-1978."),
    ] = False,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Search page-aligned chunks with the local keyword index."""

    settings = get_settings()
    response = search_keyword_index(
        index_path=settings.keyword_index_path,
        query=query,
        aliases_path=settings.person_aliases_path,
        top_k=top_k,
        document_ids=document_id,
        include_out_of_scope=include_out_of_scope,
    )
    if json_output:
        typer.echo(response.model_dump_json(indent=2))
        return
    typer.echo(
        f"intent: {response.query_intent}; query terms: {', '.join(response.query_terms)}"
    )
    if not response.hits:
        typer.echo("No matching local evidence.")
        return
    for hit in response.hits:
        section = " > ".join(hit.section_path)
        preview = hit.text.replace("\n", " ")[:220]
        typer.echo(
            f"#{hit.rank} score={hit.score:.4f} {hit.title} PDF p.{hit.pdf_page_start}"
        )
        if section:
            typer.echo(f"   {section}")
        typer.echo(f"   {preview}")


@app.command("search-vector")
def search_vector(
    query: Annotated[str, typer.Argument(help="Chinese historical research query.")],
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 10,
    document_id: Annotated[
        list[str] | None,
        typer.Option("--document-id", help="Restrict to one or more document IDs."),
    ] = None,
    include_out_of_scope: Annotated[
        bool,
        typer.Option("--include-out-of-scope", help="Include evidence outside 1921-1978."),
    ] = False,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Search local Qdrant vectors using the configured Chinese embedding model."""

    settings = get_settings()
    response = search_vector_index(
        index_path=settings.vector_index_path,
        model_cache_dir=settings.model_cache_dir / "fastembed",
        aliases_path=settings.person_aliases_path,
        query=query,
        top_k=top_k,
        document_ids=document_id,
        include_out_of_scope=include_out_of_scope,
    )
    if json_output:
        typer.echo(response.model_dump_json(indent=2))
        return
    typer.echo(f"intent: {response.query_intent}; vector model search")
    if not response.hits:
        typer.echo("No matching local evidence.")
        return
    for hit in response.hits:
        section = " > ".join(hit.section_path)
        preview = hit.text.replace("\n", " ")[:220]
        typer.echo(
            f"#{hit.rank} score={hit.score:.4f} {hit.title} PDF p.{hit.pdf_page_start}"
        )
        if section:
            typer.echo(f"   {section}")
        typer.echo(f"   {preview}")


@app.command("search-hybrid")
def search_hybrid(
    query: Annotated[str, typer.Argument(help="Chinese historical research query.")],
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=100)] = 10,
    document_id: Annotated[
        list[str] | None,
        typer.Option("--document-id", help="Restrict to one or more document IDs."),
    ] = None,
    include_out_of_scope: Annotated[
        bool,
        typer.Option("--include-out-of-scope", help="Include evidence outside 1921-1978."),
    ] = False,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Fuse local keyword and vector results with reciprocal-rank fusion."""

    settings = get_settings()
    response = search_hybrid_index(
        keyword_index_path=settings.keyword_index_path,
        vector_index_path=settings.vector_index_path,
        model_cache_dir=settings.model_cache_dir / "fastembed",
        aliases_path=settings.person_aliases_path,
        query=query,
        top_k=top_k,
        document_ids=document_id,
        include_out_of_scope=include_out_of_scope,
    )
    if json_output:
        typer.echo(response.model_dump_json(indent=2))
        return
    typer.echo(f"intent: {response.query_intent}; hybrid keyword + vector search")
    if not response.hits:
        typer.echo("No matching local evidence.")
        return
    for hit in response.hits:
        section = " > ".join(hit.section_path)
        preview = hit.text.replace("\n", " ")[:220]
        ranks = f"keyword={hit.keyword_rank or '-'}, vector={hit.vector_rank or '-'}"
        typer.echo(
            f"#{hit.rank} score={hit.score:.4f} ({ranks}) "
            f"{hit.title} PDF p.{hit.pdf_page_start}"
        )
        if section:
            typer.echo(f"   {section}")
        typer.echo(f"   {preview}")


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Reload on source changes.")] = False,
) -> None:
    """Run the local question API and single-page chat interface."""

    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "App dependencies are missing. Run: uv sync --extra app --extra vector"
        ) from exc
    uvicorn.run("history_agent.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
