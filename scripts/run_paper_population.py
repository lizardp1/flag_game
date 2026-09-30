"""Run the paper population design with a replacement OpenAI model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnd.experiment_cli import Experiment, execute, can_skip, fingerprint, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def plans(model, baseline, out, temperature=.2, reasoning_effort=None, smoke=False):
    design = json.loads((ROOT/'experiments/social/paper_population.json').read_text())
    result = []
    for case in design['baselines'][baseline]:
        spec = {**design['common'], **case, 'composition': {model: case['N']},
                'temperature': temperature, 'reasoning_effort': reasoning_effort,
                'output_root': out/f"N{case['N']}"}
        if smoke:
            spec.update(rounds=1, seeds=[case['seeds'][0]], early_stop_window=0)
        cfg = Experiment.model_validate(spec)
        for seed in cfg.seeds:
            cfg.resolve(seed)
        result.append(cfg)
        if smoke:
            break
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--baseline', choices=['gpt4o', 'gpt54'], default='gpt4o')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--temperature', type=float, default=.2)
    p.add_argument('--reasoning-effort', choices=['none'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true', help='One N=4 seed, one round; use a separate output directory')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    configs = plans(a.model, a.baseline, a.out, a.temperature, a.reasoning_effort, a.smoke)
    print(json.dumps({'baseline': a.baseline, 'trials': sum(len(c.seeds) for c in configs),
                      'configs': [c.model_dump(mode='json') for c in configs]}, indent=2))
    if a.dry_run:
        return
    source = source_fingerprint()
    for cfg in configs:
        for seed in cfg.seeds:
            can_skip(cfg.output_root/f'seed_{seed:04d}', fingerprint(cfg), source, a.resume)
    for cfg in configs:
        execute(cfg, resume=a.resume)


if __name__ == '__main__':
    main()
