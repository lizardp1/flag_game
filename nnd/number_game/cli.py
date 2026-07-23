from __future__ import annotations

import argparse
from pathlib import Path

from nnd.number_game.config import apply_overrides, load_number_game_config
from nnd.number_game.conflict import run_social_conflict_battery
from nnd.number_game.runner import run_number_game_experiment, run_pairwise_m_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-only number clue game experiments.")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--out", required=True, type=Path)
    run_parser.add_argument("--seed", default=0, type=int)
    run_parser.add_argument("--backend", default=None)
    run_parser.add_argument("--protocol", default=None)
    run_parser.add_argument("--override", action="append", default=[])
    compare_parser = subparsers.add_parser("compare-pairwise-m")
    compare_parser.add_argument("--config", required=True, type=Path)
    compare_parser.add_argument("--out", required=True, type=Path)
    compare_parser.add_argument("--start-seed", default=0, type=int)
    compare_parser.add_argument("--num-seeds", default=5, type=int)
    compare_parser.add_argument("--backend", default=None)
    compare_parser.add_argument("--m", action="append", type=int, default=[])
    compare_parser.add_argument("--override", action="append", default=[])
    conflict_parser = subparsers.add_parser("conflict-battery")
    conflict_parser.add_argument("--config", required=True, type=Path)
    conflict_parser.add_argument("--out", required=True, type=Path)
    conflict_parser.add_argument("--start-seed", default=0, type=int)
    conflict_parser.add_argument("--num-seeds", default=5, type=int)
    conflict_parser.add_argument("--backend", default=None)
    conflict_parser.add_argument("--m", action="append", type=int, default=[])
    conflict_parser.add_argument("--memory-strength", action="append", type=int, default=[])
    conflict_parser.add_argument("--ratio-total", default=None, type=int)
    conflict_parser.add_argument("--hidden-layer", action="append", type=int, default=[])
    conflict_parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.command not in ("run", "compare-pairwise-m", "conflict-battery"):
        parser.print_help()
        return

    config_obj = load_number_game_config(args.config)
    if args.override:
        config_obj = apply_overrides(config_obj, args.override)
    if args.command == "run" and args.protocol is not None:
        config_obj = config_obj.model_copy(update={"protocol": args.protocol})
    if args.backend is not None:
        config_obj = config_obj.model_copy(update={"backend": args.backend})

    if args.command == "compare-pairwise-m":
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
        result = run_pairwise_m_comparison(
            config_obj,
            out_dir=args.out,
            seeds=seeds,
            m_values=args.m or [1, 3],
        )
        print(f"Pairwise m comparison complete. Output saved to {args.out}")
        for row in result["summary"]:
            print(
                "m={m}: accuracy={mean_final_accuracy:.3f}, consensus_correct={consensus_correct_rate:.3f}, valid={mean_valid_rate:.3f}".format(
                    **row
                )
        )
        return

    if args.command == "conflict-battery":
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
        result = run_social_conflict_battery(
            config_obj,
            out_dir=args.out,
            seeds=seeds,
            m_values=args.m or [1, 3],
            memory_strengths=args.memory_strength or None,
            hidden_state_layers=args.hidden_layer or None,
            ratio_total=args.ratio_total,
        )
        print(f"Social conflict battery complete. Output saved to {args.out}")
        for row in result["summary"]:
            if row["memory_relation"] != "conflict":
                continue
            print(
                "m={m}, memory={memory_strength}: social_over_private={social_over_private_rate:.3f}, private_resists={private_resists_social_rate:.3f}, satisfies_clue={satisfies_private_clue_rate:.3f}".format(
                    **row
                )
            )
        return

    result = run_number_game_experiment(config_obj, out_dir=args.out, seed=args.seed)
    print(f"Run complete. Output saved to {args.out}")
    print(f"Protocol: {result['summary']['protocol']}")
    print(f"Truth number: {result['summary']['truth_number']}")
    print(f"Final consensus: {result['summary']['final_consensus_number']}")
    print(f"Final accuracy: {result['summary']['final_accuracy']:.3f}")


if __name__ == "__main__":
    main()
