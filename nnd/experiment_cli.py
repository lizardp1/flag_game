"""One public interface for Flag Game protocols; legacy CLIs remain available."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Literal

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

app = typer.Typer(no_args_is_help=True)


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: Literal["pairwise", "broadcast", "manager"]
    backend: Literal["scripted", "openai", "anthropic"] = "scripted"
    N: int = Field(default=8, ge=1)
    composition: dict[str, StrictInt] = Field(default_factory=lambda: {"gpt-4o": 8})
    manager_model: str = "gpt-4o"
    social_evidence_alpha: float = Field(default=0.5, ge=0, le=1)
    social_guidance_enabled: bool = False
    message_bandwidth: Literal[1, 2, 3] = 3
    memory_capacity: int = Field(default=8, ge=0)
    rounds: int = Field(default=10, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    randomize_model_slots: bool = True
    country_pool: str = "stripe_plus_real_triangle_28"
    fixed_truth_country: str | None = None
    canvas_width: int = 24
    canvas_height: int = 16
    tile_width: int = 6
    tile_height: int = 4
    render_scale: int = Field(default=25, ge=1)
    image_detail: Literal["auto", "low", "high", "original"] = "high"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = Field(default=250, ge=1)
    consensus_threshold: float = Field(default=0.85, gt=0, le=1)
    polarization_threshold: float = Field(default=0.25, gt=0, le=1)
    early_stop_window: int = Field(default=0, ge=0)
    workers: int = Field(default=1, ge=1)
    make_plots: bool = False
    save_crop_images: bool = True
    output_root: Path = Path("runs/social")

    @model_validator(mode="after")
    def validate_design(self):
        if any(isinstance(v, bool) or v < 1 for v in self.composition.values()):
            raise ValueError("composition counts must be positive integers")
        if sum(self.composition.values()) != self.N:
            raise ValueError("composition counts must sum to N (manager is additional)")
        if len(set(self.seeds)) != len(self.seeds) or any(s < 0 for s in self.seeds):
            raise ValueError("seeds must be unique nonnegative integers")
        if self.protocol == "broadcast" and len(self.composition) > 1 and not self.randomize_model_slots:
            raise ValueError("Anonymous mixed-model broadcast requires randomize_model_slots=true")
        if self.protocol in ("pairwise", "broadcast") and self.N < 2:
            raise ValueError("Population protocols require N >= 2; use a single-agent probe for N=1")
        if self.backend == "anthropic" and self.protocol != "pairwise":
            raise ValueError("Anthropic is currently supported by pairwise/probes only")
        models = list(self.composition) + ([self.manager_model] if self.protocol == "manager" else [])
        if any(not m.strip() for m in models):
            raise ValueError("model names must not be empty")
        if "gpt-5.4" in models:
            raise ValueError("Use the pinned model gpt-5.4-2026-03-05 for new runs")
        if self.backend == "openai" and any(not m.startswith(("gpt-", "o1", "o3", "o4")) for m in models):
            raise ValueError("OpenAI runs require OpenAI models; cross-provider mixtures are unsupported")
        if self.backend == "anthropic" and any(not m.startswith("claude-") for m in models):
            raise ValueError("Anthropic runs require Claude models")
        return self

    def resolve(self, seed: int):
        from nnd.flag_game.config import FlagGameConfig
        from nnd.flag_game_broadcast.config import BroadcastFlagGameConfig
        from nnd.flag_game_org.config import OrgFlagGameConfig
        models = [name for name, count in self.composition.items() for _ in range(count)]
        # A separate RNG prevents model allocation from changing crops/schedules.
        if self.randomize_model_slots and self.protocol != "broadcast":
            random.Random(seed).shuffle(models)
        common = dict(backend=self.backend, model=models[0], agent_models=models, N=self.N,
                      H=self.memory_capacity, interaction_m=self.message_bandwidth,
                      social_susceptibility=self.social_evidence_alpha,
                      prompt_social_susceptibility=self.social_guidance_enabled,
                      country_pool=self.country_pool, fixed_truth_country=self.fixed_truth_country,
                      canvas_width=self.canvas_width, canvas_height=self.canvas_height,
                      tile_width=self.tile_width, tile_height=self.tile_height, render_scale=self.render_scale,
                      image_detail=self.image_detail, temperature=self.temperature, top_p=self.top_p,
                      max_tokens=self.max_tokens, consensus_threshold=self.consensus_threshold,
                      polarization_threshold=self.polarization_threshold,
                      output=dict(make_plots=self.make_plots, save_crop_images=self.save_crop_images))
        if self.protocol == "pairwise":
            common["output"].update(include_memory_snapshots=True, include_prompt_audit=True)
            return FlagGameConfig(**common, T=self.rounds*self.N, probe_every=self.N,
                                  early_stop_probe_window=self.early_stop_window, probe_workers=self.workers)
        if self.protocol == "broadcast":
            return BroadcastFlagGameConfig(**common, rounds=self.rounds, agent_workers=self.workers,
                                           randomize_agent_model_slots=self.randomize_model_slots,
                                           max_influential_agents=min(3,self.N-1), early_stop_round_window=self.early_stop_window)
        common["agent_models"] = [self.manager_model] + models
        return OrgFlagGameConfig(**common, aggregator_agent_id=0, rounds=self.rounds,
                                 agent_workers=self.workers, early_stop_round_window=self.early_stop_window)


def load(path: Path, overrides: list[str] | None = None) -> Experiment:
    data = yaml.safe_load(path.read_text())
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("--set expects key=value")
        key,value = item.split("=",1)
        data[key] = yaml.safe_load(value)
    config = Experiment.model_validate(data)
    for seed in config.seeds:
        config.resolve(seed)
    return config


def prompt_examples(config: Experiment) -> dict[str,str]:
    from nnd.flag_game.catalog import COUNTRY_POOLS
    from nnd.flag_game import prompts as p
    from nnd.flag_game_broadcast import prompts as b
    from nnd.flag_game_org import prompts as o
    countries = list(COUNTRY_POOLS[config.country_pool])
    args = dict(countries=countries, memory_lines=[], m=config.message_bandwidth)
    guidance = dict(social_susceptibility=config.social_evidence_alpha,
                    prompt_social_susceptibility=config.social_guidance_enabled)
    if config.protocol == "pairwise":
        return dict(system=p.system_prompt(), interaction=p.interaction_text(**args, **guidance),
                    probe=p.probe_text(**args, **guidance), retry=p.interaction_retry_text(countries=countries,m=config.message_bandwidth,error_text="invalid schema"))
    if config.protocol == "broadcast":
        return dict(system=b.system_prompt(), statement=b.statement_text(**args),
                    decision=b.decision_text(**args,**guidance,round_broadcast_lines=[], max_influential_agents=min(3,config.N-1)),
                    statement_retry=b.statement_retry_text(countries=countries,m=config.message_bandwidth,error_text="invalid schema"),
                    decision_retry=b.decision_retry_text(countries=countries,m=config.message_bandwidth,max_influential_agents=min(3,config.N-1),error_text="invalid schema"))
    return dict(observer_system=o.observer_system_prompt(), manager_system=o.manager_system_prompt(),
                observer=o.observer_statement_text(**args,**guidance),
                manager=o.aggregator_decision_text(**args,**guidance,observer_statement_lines=[]),
                observer_retry=o.observer_statement_retry_text(countries=countries,m=config.message_bandwidth,error_text="invalid schema"),
                manager_retry=o.aggregator_decision_retry_text(countries=countries,m=config.message_bandwidth,error_text="invalid schema"))


def fingerprint(config: Experiment) -> str:
    data=config.model_dump(mode="json", exclude={"output_root", "seeds"})
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def source_fingerprint() -> str:
    root=Path(__file__).parent
    digest=hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    return digest.hexdigest()


def can_skip(out: Path, digest: str, source_hash: str, resume: bool) -> bool:
    if not out.exists():
        return False
    done=out/'experiment.json'
    if resume and done.exists():
        previous=json.loads(done.read_text())
        if previous.get('status')=='complete' and previous.get('configuration_hash')==digest and previous.get('source_hash')==source_hash:
            return True
    raise ValueError(f"Refusing to overwrite {out}; choose a fresh output_root. --resume only skips identical completed trials.")


def execute(config: Experiment, resume: bool=False) -> None:
    from nnd.flag_game.runner import run_flag_game_experiment
    from nnd.flag_game_broadcast.runner import run_broadcast_flag_game_experiment
    from nnd.flag_game_org.runner import run_org_flag_game_experiment
    runners=dict(pairwise=run_flag_game_experiment,broadcast=run_broadcast_flag_game_experiment,manager=run_org_flag_game_experiment)
    digest=fingerprint(config); source_hash=source_fingerprint()
    plan=[]
    for seed in config.seeds:
        out=config.output_root/f"seed_{seed:04d}"
        if can_skip(out,digest,source_hash,resume):
            continue
        plan.append((seed,out,config.resolve(seed)))
    for seed,out,resolved in plan:
        out.mkdir(parents=True,exist_ok=False)
        record=dict(status='running', protocol=config.protocol, seed=seed,
                    prompt_contract=f'{config.protocol}-public-v1',configuration_hash=digest,source_hash=source_hash,
                    config=config.model_dump(mode='json'), resolved=resolved.model_dump(mode='json'),
                    scripted_backend_is_behavioral_evidence=False)
        import importlib.metadata
        record['dependencies']={name: importlib.metadata.version(name) for name in ['numpy','pandas','pydantic','openai','anthropic']}
        def save(): (out/'experiment.json').write_text(json.dumps(record,indent=2)+'\n')
        save();(out/'prompt_examples.json').write_text(json.dumps(prompt_examples(config),indent=2)+'\n')
        try:
            result=runners[config.protocol](resolved,out_dir=out,seed=seed)
            record.update(status='complete', summary=result['summary'])
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__);save();raise
        save();typer.echo(f"Completed {config.protocol}: {out}")


@app.command()
def run(config: Path=typer.Option(...), set: list[str]=typer.Option([],"--set"),
        dry_run: bool=False, resume: bool=False):
    """Run social trials. --dry-run validates and prints resolved settings without calls."""
    cfg=load(config,set)
    if dry_run:
        typer.echo(json.dumps({"config":cfg.model_dump(mode='json'),"resolved":cfg.resolve(cfg.seeds[0]).model_dump(mode='json')},indent=2));return
    execute(cfg,resume)


@app.command()
def prompts(config: Path=typer.Option(...), set: list[str]=typer.Option([],"--set")):
    """Render all stages and retry examples; never calls a model."""
    typer.echo(json.dumps(prompt_examples(load(config,set)),indent=2))


@app.command()
def sweep(config: Path=typer.Option(...), dry_run: bool=False, resume: bool=False):
    """Run a declarative Cartesian sweep; validate every cell before any calls."""
    data=yaml.safe_load(config.read_text())
    if set(data)-{'base','grid','cases','output_root'}:raise ValueError('Unknown sweep keys')
    base_path=config.parent/data['base'];base=yaml.safe_load(base_path.read_text())
    if ('grid' in data) == ('cases' in data):raise ValueError('Choose exactly one of grid or cases')
    if 'grid' in data:
        grid=data['grid'];keys=list(grid)
        if not keys or any(not isinstance(v,list) or not v for v in grid.values()):raise ValueError('grid values must be nonempty lists')
        cells=[dict(zip(keys,values)) for values in itertools.product(*(grid[k] for k in keys))]
    else:
        cells=data['cases']
        if not cells or any(not isinstance(c,dict) for c in cells):raise ValueError('cases must be a nonempty list of mappings')
    plans=[];seen=set()
    for cell in cells:
        spec={**base,**cell}
        cfg=Experiment.model_validate(spec)
        for seed in cfg.seeds:cfg.resolve(seed)
        h=fingerprint(cfg)
        if h in seen:raise ValueError('Duplicate sweep cells')
        seen.add(h);cfg.output_root=Path(data['output_root'])/f'{cfg.protocol}_{h[:12]}'
        plans.append(cfg)
    if dry_run:
        typer.echo(json.dumps([p.model_dump(mode='json') for p in plans],indent=2));return
    # Preflight paths across the entire sweep to avoid partial execution on collision.
    source_hash=source_fingerprint()
    for cfg in plans:
        for seed in cfg.seeds:
            path=cfg.output_root/f'seed_{seed:04d}'
            can_skip(path,fingerprint(cfg),source_hash,resume)
    for cfg in plans:execute(cfg,resume)


@app.command(context_settings={"allow_extra_args":True,"ignore_unknown_options":True,"help_option_names":[]})
def probe(ctx: typer.Context, kind: Literal["memory","vision"]):
    """Run an isolated-agent experiment. Remaining arguments go to its dedicated runner."""
    import sys
    module='nnd.probes.'+kind
    result=subprocess.run([sys.executable,'-m',module,*ctx.args],check=False)
    raise typer.Exit(result.returncode)


@app.command()
def analyze(runs: Path=typer.Option(...), out: Path=typer.Option(...)):
    """Aggregate completed social trials into stable CSV tables; no inference."""
    import pandas as pd
    rows=[]
    for path in sorted(runs.rglob("experiment.json")):
        data=json.loads(path.read_text())
        if data.get("status")!="complete":continue
        config=data["config"];summary=data["summary"]
        rows.append(dict(protocol=config["protocol"],N=config["N"],
                         alpha=config["social_evidence_alpha"],guidance=config["social_guidance_enabled"],
                         bandwidth=config["message_bandwidth"],configuration_hash=data["configuration_hash"],
                         composition=json.dumps(config["composition"],sort_keys=True),
                         seed=data["seed"],backend=config["backend"],
                         final_accuracy=summary["final_accuracy"],
                         outcome=summary.get("final_outcome"),source=str(path)))
    if not rows:raise ValueError("No completed social trials found")
    frame=pd.DataFrame(rows)
    if frame.duplicated(["configuration_hash","seed"]).any():
        raise ValueError("Duplicate configuration/seed trials; select one cohort explicitly")
    keys=["protocol","N","alpha","guidance","bandwidth","composition","backend","configuration_hash"]
    grouped=frame.groupby(keys,dropna=False)["final_accuracy"].agg(count="count",mean="mean",std="std").reset_index()
    grouped["sem"]=grouped["std"]/(grouped["count"]**.5)
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/"trials.csv",index=False);grouped.to_csv(out/"summary.csv",index=False)
    (out/"sources.json").write_text(json.dumps({r["source"]:hashlib.sha256(Path(r["source"]).read_bytes()).hexdigest() for r in rows},indent=2)+"\n")
    typer.echo(f"Wrote {len(frame)} trials and {len(grouped)} conditions to {out}")


@app.command()
def plot(summary: Path=typer.Option(...), out: Path=typer.Option(...), font_size: float=10):
    """Redraw an accuracy chart from a summary table; never starts a run."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame=pd.read_csv(summary)
    with plt.rc_context({"font.size":font_size}):
        fig,ax=plt.subplots(figsize=(8,max(3,len(frame)*.3)))
        positions=list(range(len(frame)))
        ax.barh(positions,frame["mean"],xerr=frame["sem"].fillna(0))
        labels=[f"{r.protocol}, N={r.N}, m={r.bandwidth}, alpha={r.alpha}, guidance={r.guidance}, {r.configuration_hash[:6]}" for r in frame.itertuples()]
        ax.set_yticks(positions,labels=labels);ax.set_xlim(0,1)
        ax.set_xlabel("Final accuracy (population mean or manager endpoint, per protocol)")
        ax.set_title("Scripted pipeline check" if set(frame.backend)=={"scripted"} else "Flag Game")
        fig.tight_layout();out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out);plt.close(fig)
    typer.echo(f"Updated {out}")


if __name__=='__main__':app()
