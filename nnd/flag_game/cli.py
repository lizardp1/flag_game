from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from nnd.flag_game.config import apply_overrides, load_flag_game_config
from nnd.flag_game.model_mix import run_flag_game_model_mix_comparison
from nnd.flag_game.runner import (
    choose_default_backend,
    run_flag_game_batch,
    run_flag_game_experiment,
    run_flag_game_sweep,
)


app = typer.Typer(no_args_is_help=True)
console = Console()


def _parse_tile_specs(values: list[str]) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    for value in values:
        text = value.strip().lower()
        if "x" not in text:
            raise typer.BadParameter(f"Tile spec must look like WIDTHxHEIGHT, got: {value}")
        width_text, height_text = text.split("x", 1)
        try:
            width = int(width_text)
            height = int(height_text)
        except ValueError as exc:
            raise typer.BadParameter(f"Tile spec must look like WIDTHxHEIGHT, got: {value}") from exc
        if width < 1 or height < 1:
            raise typer.BadParameter(f"Tile spec dimensions must be >= 1, got: {value}")
        specs.append((width, height))
    return specs


@app.command()
def run(
    config: Path = typer.Option(..., "--config"),
    out: Path = typer.Option(..., "--out"),
    seed: int = typer.Option(0, "--seed"),
    backend: str | None = typer.Option(None, "--backend"),
    probe_workers: int | None = typer.Option(None, "--probe-workers"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    config_obj = load_flag_game_config(config)
    if override:
        config_obj = apply_overrides(config_obj, override)
    if probe_workers is not None:
        config_obj = config_obj.model_copy(update={"probe_workers": probe_workers})
    if backend is not None:
        config_obj = config_obj.model_copy(update={"backend": backend})
    else:
        config_obj = config_obj.model_copy(update={"backend": choose_default_backend()})
    result = run_flag_game_experiment(config_obj, out_dir=out, seed=seed)
    console.print(f"Run complete. Output saved to {out}")
    console.print(f"Truth country: {result['summary']['truth_country']}")
    console.print(f"Final outcome: {result['summary']['final_outcome']}")
    console.print(f"Final accuracy: {result['summary']['final_accuracy']:.3f}")


@app.command()
def batch(
    config: Path = typer.Option(..., "--config"),
    out: Path = typer.Option(..., "--out"),
    start_seed: int = typer.Option(0, "--start-seed"),
    num_seeds: int = typer.Option(5, "--num-seeds"),
    backend: str | None = typer.Option(None, "--backend"),
    probe_workers: int | None = typer.Option(None, "--probe-workers"),
    seed_workers: int | None = typer.Option(None, "--seed-workers"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    config_obj = load_flag_game_config(config)
    if override:
        config_obj = apply_overrides(config_obj, override)
    if probe_workers is not None:
        config_obj = config_obj.model_copy(update={"probe_workers": probe_workers})
    if seed_workers is not None:
        config_obj = config_obj.model_copy(update={"seed_workers": seed_workers})
    if backend is not None:
        config_obj = config_obj.model_copy(update={"backend": backend})
    else:
        config_obj = config_obj.model_copy(update={"backend": choose_default_backend()})
    seeds = list(range(start_seed, start_seed + num_seeds))
    results = run_flag_game_batch(config_obj, out_dir=out, seeds=seeds)
    exact_rate = sum(1 for row in results if row["summary"]["final_consensus_correct"]) / float(len(results))
    console.print(f"Batch complete. Output saved to {out}")
    console.print(f"Correct-consensus rate: {exact_rate:.3f}")


@app.command()
def sweep(
    config: Path = typer.Option(..., "--config"),
    out: Path = typer.Option(..., "--out"),
    n_value: list[int] = typer.Option(..., "--n", help="Repeat --n for each population size to include."),
    m_value: list[int] = typer.Option(..., "--m", help="Repeat --m for each interaction bandwidth to include."),
    tile: list[str] = typer.Option([], "--tile", help="Repeat --tile WIDTHxHEIGHT to sweep crop sizes, e.g. 3x4."),
    start_seed: int = typer.Option(0, "--start-seed"),
    num_seeds: int = typer.Option(5, "--num-seeds"),
    rounds: int | None = typer.Option(None, "--rounds", help="If set, use T = rounds * N for each condition."),
    scale_t_with_n: bool = typer.Option(True, "--scale-t-with-n/--fixed-t"),
    backend: str | None = typer.Option(None, "--backend"),
    probe_workers: int | None = typer.Option(None, "--probe-workers"),
    seed_workers: int | None = typer.Option(None, "--seed-workers"),
    condition_workers: int | None = typer.Option(None, "--condition-workers"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    config_obj = load_flag_game_config(config)
    if override:
        config_obj = apply_overrides(config_obj, override)
    if probe_workers is not None:
        config_obj = config_obj.model_copy(update={"probe_workers": probe_workers})
    if seed_workers is not None:
        config_obj = config_obj.model_copy(update={"seed_workers": seed_workers})
    if condition_workers is not None:
        config_obj = config_obj.model_copy(update={"condition_workers": condition_workers})
    if backend is not None:
        config_obj = config_obj.model_copy(update={"backend": backend})
    else:
        config_obj = config_obj.model_copy(update={"backend": choose_default_backend()})

    seeds = list(range(start_seed, start_seed + num_seeds))
    tile_sizes = _parse_tile_specs(tile) if tile else None
    result = run_flag_game_sweep(
        config_obj,
        out_dir=out,
        n_values=n_value,
        m_values=m_value,
        seeds=seeds,
        tile_sizes=tile_sizes,
        scale_t_with_n=scale_t_with_n,
        rounds=rounds,
        make_plots=config_obj.output.make_plots,
    )
    console.print(f"Sweep complete. Output saved to {out}")
    console.print(f"Conditions run: {len(result['summary'])}")
    console.print("Wrote sweep summaries:")
    console.print(f"- {out / 'sweep_condition_results.csv'}")
    console.print(f"- {out / 'sweep_summary.csv'}")
    if result["skipped_conditions"]:
        console.print(f"- {out / 'sweep_skipped_conditions.csv'}")


@app.command()
def model_mix(
    config: Path = typer.Option(..., "--config"),
    out: Path = typer.Option(..., "--out"),
    baseline_model: str = typer.Option("gpt-4o", "--baseline-model"),
    boosted_model: str = typer.Option("gpt-5.4", "--boosted-model"),
    boosted_agent_id: list[int] = typer.Option(
        [],
        "--boosted-agent-id",
        help="Repeat to choose which agent ids use the boosted model in the mixed condition.",
    ),
    start_seed: int = typer.Option(0, "--start-seed"),
    num_seeds: int = typer.Option(5, "--num-seeds"),
    include_pure_controls: bool = typer.Option(
        True,
        "--include-pure-controls/--skip-pure-controls",
        help="Include or skip the all-baseline and all-boosted control conditions.",
    ),
    include_mixed_condition: bool = typer.Option(
        True,
        "--include-mixed-condition/--skip-mixed-condition",
        help="Include or skip the mixed one-or-more-boosted-agent condition.",
    ),
    backend: str | None = typer.Option(None, "--backend"),
    probe_workers: int | None = typer.Option(None, "--probe-workers"),
    seed_workers: int | None = typer.Option(None, "--seed-workers"),
    condition_workers: int | None = typer.Option(
        None,
        "--condition-workers",
        help="Number of model-mix conditions to run concurrently.",
    ),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    config_obj = load_flag_game_config(config)
    if override:
        config_obj = apply_overrides(config_obj, override)
    if probe_workers is not None:
        config_obj = config_obj.model_copy(update={"probe_workers": probe_workers})
    if seed_workers is not None:
        config_obj = config_obj.model_copy(update={"seed_workers": seed_workers})
    if condition_workers is not None:
        config_obj = config_obj.model_copy(update={"condition_workers": condition_workers})
    if backend is not None:
        config_obj = config_obj.model_copy(update={"backend": backend})
    else:
        config_obj = config_obj.model_copy(update={"backend": "openai"})

    seeds = list(range(start_seed, start_seed + num_seeds))
    active_boosted_ids = boosted_agent_id or [0]
    result = run_flag_game_model_mix_comparison(
        config_obj,
        out_dir=out,
        seeds=seeds,
        baseline_model=baseline_model,
        boosted_model=boosted_model,
        boosted_agent_ids=active_boosted_ids,
        include_pure_controls=include_pure_controls,
        include_mixed_condition=include_mixed_condition,
        condition_workers=config_obj.condition_workers,
    )
    console.print(f"Model-mix comparison complete. Output saved to {out}")
    console.print(f"Conditions run: {len(result['summary'])}")
    console.print("Wrote comparison summaries:")
    console.print(f"- {out / 'model_mix_summary.csv'}")
    console.print(f"- {out / 'model_mix_condition_results.csv'}")
    console.print(f"- {out / 'model_mix_boosted_agent_diagnostics.csv'}")
    console.print(f"- {out / 'model_mix_paired_summary.csv'}")
    console.print(f"- {out / 'model_mix_report.md'}")


if __name__ == "__main__":
    app()
