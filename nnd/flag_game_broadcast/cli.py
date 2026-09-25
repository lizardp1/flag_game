from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from nnd.flag_game.runner import choose_default_backend
from nnd.flag_game_broadcast.config import (
    apply_overrides,
    load_broadcast_flag_game_config,
)
from nnd.flag_game_broadcast.runner import (
    run_broadcast_flag_game_batch,
    run_broadcast_flag_game_experiment,
    run_broadcast_flag_game_mix_sweep,
)


app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., "--config"),
    out: Path = typer.Option(..., "--out"),
    seed: int = typer.Option(0, "--seed"),
    backend: str | None = typer.Option(None, "--backend"),
    agent_workers: int | None = typer.Option(None, "--agent-workers"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    config_obj = load_broadcast_flag_game_config(config)
    if override:
        config_obj = apply_overrides(config_obj, override)
    if agent_workers is not None:
        config_obj = config_obj.model_copy(update={"agent_workers": agent_workers})
    if backend is not None:
        config_obj = config_obj.model_copy(update={"backend": backend})
    else:
        config_obj = config_obj.model_copy(update={"backend": choose_default_backend()})
    result = run_broadcast_flag_game_experiment(config_obj, out_dir=out, seed=seed)
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
    agent_workers: int | None = typer.Option(None, "--agent-workers"),
    seed_workers: int | None = typer.Option(None, "--seed-workers"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    config_obj = load_broadcast_flag_game_config(config)
    if override:
        config_obj = apply_overrides(config_obj, override)
    if agent_workers is not None:
        config_obj = config_obj.model_copy(update={"agent_workers": agent_workers})
    if seed_workers is not None:
        config_obj = config_obj.model_copy(update={"seed_workers": seed_workers})
    if backend is not None:
        config_obj = config_obj.model_copy(update={"backend": backend})
    else:
        config_obj = config_obj.model_copy(update={"backend": choose_default_backend()})
    seeds = list(range(start_seed, start_seed + num_seeds))
    results = run_broadcast_flag_game_batch(config_obj, out_dir=out, seeds=seeds)
    exact_rate = sum(1 for row in results if row["summary"]["final_consensus_correct"]) / float(len(results))
    console.print(f"Batch complete. Output saved to {out}")
    console.print(f"Correct-consensus rate: {exact_rate:.3f}")


@app.command("mix-sweep")
def mix_sweep(
    config: Path = typer.Option(..., "--config"),
    out: Path = typer.Option(..., "--out"),
    comparison_model: str = typer.Option("gpt-4o", "--comparison-model"),
    prestige_model: str = typer.Option("gpt-5.4", "--prestige-model"),
    prestige_count: list[int] = typer.Option(
        ...,
        "--prestige-count",
        help="Repeat --prestige-count to choose how many prestige-model agents are in each condition.",
    ),
    start_seed: int = typer.Option(0, "--start-seed"),
    num_seeds: int = typer.Option(5, "--num-seeds"),
    include_pure_controls: bool = typer.Option(
        True,
        "--include-pure-controls/--skip-pure-controls",
    ),
    backend: str | None = typer.Option(None, "--backend"),
    agent_workers: int | None = typer.Option(None, "--agent-workers"),
    seed_workers: int | None = typer.Option(None, "--seed-workers"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    config_obj = load_broadcast_flag_game_config(config)
    if override:
        config_obj = apply_overrides(config_obj, override)
    if agent_workers is not None:
        config_obj = config_obj.model_copy(update={"agent_workers": agent_workers})
    if seed_workers is not None:
        config_obj = config_obj.model_copy(update={"seed_workers": seed_workers})
    if backend is not None:
        config_obj = config_obj.model_copy(update={"backend": backend})
    else:
        config_obj = config_obj.model_copy(update={"backend": "openai"})
    config_obj = config_obj.model_copy(
        update={
            "comparison_model_label": comparison_model,
            "prestige_model_label": prestige_model,
        }
    )

    seeds = list(range(start_seed, start_seed + num_seeds))
    result = run_broadcast_flag_game_mix_sweep(
        config_obj,
        out_dir=out,
        seeds=seeds,
        comparison_model=comparison_model,
        prestige_model=prestige_model,
        prestige_counts=prestige_count,
        include_pure_controls=include_pure_controls,
    )
    console.print(f"Mix sweep complete. Output saved to {out}")
    console.print(f"Conditions run: {len(result['summary'])}")
    console.print("Wrote comparison summaries:")
    console.print(f"- {out / 'mix_summary.csv'}")
    console.print(f"- {out / 'mix_condition_results.csv'}")
    console.print(f"- {out / 'mix_report.md'}")


if __name__ == "__main__":
    app()
