# Hugging Face inventory: 9x25dillon

Inspected public repository metadata through the Hub API on 2026-09-20. The live API
listed nine model repositories and one dataset repository. The cached public profile
page listed eight models, so the saved API inventory is the source for this build.

Full local inventory: `artifacts/huggingface/catalog.json`. It includes immutable
commit IDs, file names, license metadata when supplied, and gate status.

| Repository | Finding relevant to this build |
| --- | --- |
| `dual-llm-wavecaster-system` | Contains the 70-record `second_llm_training_prompts.jsonl` |
| `advanced-tokenizer-system` | Contains small matrix/emergent JSONL datasets |
| `LFM2-8B-A1B-Dimensional-Entanglement` | Contains emergent training JSONL; no complete model weights listed |
| `LiMp-Pipeline-Integration-System` | Source, integration, training and model-card files |
| `newthought-quantum-coherence` | Additional experimental source and documents |
| `klancker` (dataset) | Only `.gitattributes` and `README.md` listed; no dataset payload |

No complete `.safetensors`, `.pt`, or `.gguf` checkpoint was found in these model
file inventories. One older repository lists incomplete `.crdownload` artifacts.
Repository names and model cards alone are not evidence of trained model weights.

The selected prompts were downloaded from commit
`003824705ff00533af8b6c56c3793885e736a875` of
[`9x25dillon/dual-llm-wavecaster-system`](https://huggingface.co/9x25dillon/dual-llm-wavecaster-system).
The 170,665-byte file has SHA-256
`ae5892f890e451ee294d273e93cef63e045bfcf1e5195eb63ebf7a685a4e2304`.
It is byte-for-byte identical to local `LiMp/second_llm_training_prompts.jsonl`.
It has not been added a second time to the training corpus.

The inventory also includes shell-history/configuration and wallet-named files in
some public repositories. Their contents were not opened or imported. Review those
public file listings separately before treating an entire repository as a corpus.

To repeat the metadata review:

```bash
python -m auric hub-catalog --author 9x25dillon --output artifacts/huggingface/new-catalog.json
```

Fetch an explicitly chosen small data file with `hub-fetch`; use the full immutable
revision from the inventory. Downloads produce a provenance receipt and are not
automatically trained on. Gated content requires a separate authenticated workflow;
the current reader uses public access only. Nothing was uploaded or published.
