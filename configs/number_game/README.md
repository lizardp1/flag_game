# Number Game Configs

There are only three active configs.

- `local_qwen3_1_7b.yaml`: small local smoke runs on Qwen3-1.7B through Hugging Face Transformers.
- `runpod_qwen3.yaml`: RunPod template for Qwen3 sweeps; override `model`, `N`, `H`, `T`, or `protocol` at the command line.
- `runpod_kimi_k2_endpoint.yaml`: Kimi K2 template for an OpenAI-compatible vLLM/SGLang endpoint.

All active configs use:

```yaml
min_number: 1
max_number: 100
prompt_number_range: true
prompt_social_susceptibility: false
early_stop_window: 5
```

Use CLI overrides instead of adding one config per model/protocol:

```bash
--override model=Qwen/Qwen3-4B
--override N=16
--override H=8
--override protocol=broadcast
```
