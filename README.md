# Hash_Generate

Hash_Generate provides model hashing and tamper localization tools for convolutional neural networks. The current GitHub release focuses on **TLH**, including hash generation, model tampering, block-level comparison, and localization accuracy evaluation.

> **PIH code is currently being organized.** The PIH implementation and its experiment scripts will be added in a future update.

## Features

- Generate TLH from a model checkpoint using block means and a chaotic map.
- Generate a baseline TLH directly from block means, without chaotic processing.
- Apply block-wise tampering and record the affected blocks as ground truth.
- Compare original and modified TLH outputs and calculate localization accuracy `Rt`.
- Run batch experiments and export comparison tables and optional plots.

Supported model names: `resnet18`, `resnet32`, `resnet56`, `resnet110`, `vgg16`, `densenet121`, `mnist`, and `lenet`. Here, `mnist` refers to MNIST-Net.

## Requirements

- Python 3.10 or later
- PyTorch
- NumPy

Install the project dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The workspace dependency list also includes SciPy, torchvision, PyYAML, Matplotlib, and pandas. Plotting uses Matplotlib and pandas. Install a PyTorch build appropriate for your environment if CUDA is required.

## Model Checkpoints

Prepare a checkpoint compatible with the selected model architecture in `src/models/`. The examples below assume a 10-class ResNet18 checkpoint at:

```text
checkpoints/resnet18.pth
```

Replace this path with your own checkpoint path. Supported checkpoint formats include a plain state dictionary or a dictionary containing a `state_dict` or `model` entry. The loader removes a leading `module.` prefix from parameter names.

The basic TLH workflow reads model parameters and does not require an image dataset. Model weights, datasets, and generated tampered checkpoints are not part of the intended source-code release.

## Quick Start

Run all commands from the repository root. Examples use CPU, the default device, and forward-slash paths that also work in PowerShell.

### 1. Generate the Original TLH

```bash
python scripts/generate_tlh.py --model resnet18 --checkpoint checkpoints/resnet18.pth --output artifacts/hashes/tlh/resnet18_original.json --legacy-txt artifacts/hashes/tlh/resnet18_original.txt
```

The JSON output contains the method, block count, block means, processed values, and encoded TLH. The optional text output stores one encoded block per line.

### 2. Generate a Tampered Checkpoint

The following example adds Gaussian noise to 10% of the parameter blocks:

```bash
python scripts/generate_tampered.py --model resnet18 --checkpoint checkpoints/resnet18.pth --tamper-rate 0.1 --attack gaussian_noise --noise-std 0.1 --seed 42 --output artifacts/models/tampered/resnet18_r10.pth --log-output artifacts/logs/resnet18_r10_tamper.json --metadata-output artifacts/logs/resnet18_r10_metadata.json
```

`--tamper-rate` specifies the fraction of blocks selected for tampering. Other supported attacks are `sign_flip`, `weight_scaling`, `weight_zeroing`, `random_reinit`, `weight_permutation`, and `quantization`. Use `--help` for their parameters.

### 3. Generate the Tampered Model's TLH

```bash
python scripts/generate_tlh.py --model resnet18 --checkpoint artifacts/models/tampered/resnet18_r10.pth --output artifacts/hashes/tlh/resnet18_r10.json --legacy-txt artifacts/hashes/tlh/resnet18_r10.txt
```

### 4. Evaluate Tamper Localization

```bash
python scripts/calculate_tlh_rt.py --original-tlh artifacts/hashes/tlh/resnet18_original.json --tampered-tlh artifacts/hashes/tlh/resnet18_r10.json --tamper-log artifacts/logs/resnet18_r10_tamper.json --diff-output artifacts/logs/resnet18_r10_diff.json --summary-output artifacts/logs/resnet18_r10_rt.json
```

The script compares the encoded TLH block by block and reports:

- Ground-truth tampered block count
- Detected changed block count
- Correctly detected tampered block count
- False-positive block count
- Localization accuracy `Rt`

`Rt` is the number of correctly detected tampered blocks divided by the number of ground-truth tampered blocks. False positives are reported separately; `Rt` alone does not measure them.

Use the same model architecture, block count, TLH method, and chaos parameters for the original and modified models. The tamper log must correspond to the same block partition.

## Baseline Without Chaos

To generate TLH directly from block means, use `--method baseline`:

```bash
python scripts/generate_tlh.py --model resnet18 --checkpoint checkpoints/resnet18.pth --method baseline --output artifacts/hashes/tlh_baseline/resnet18_original.json
```

Generate the modified model's baseline TLH with the same option, then evaluate the pair using `calculate_tlh_rt.py`. The baseline implementation is included in `src/hashing/tlh.py`.

## Default Parameters

| Parameter | Default |
| --- | --- |
| Blocks for ResNet, VGG16, and DenseNet121 | 450 |
| Blocks for MNIST-Net and LeNet | 100 |
| Chaotic map parameter `mu` | 0.2 |
| Chaotic map parameter `k` | 2.0 |
| Chaotic map iterations | 100 |
| Encoded block length | 17 bits: one sign bit and four 4-bit decimal digits |
| Gaussian tampering noise standard deviation | 0.1 |
| Tampering random seed | 42 |
| Device | CPU |

Pass `--blocks` to set an explicit partition size. TLH generation also accepts `--mu`, `--k`, and `--iterations`. These options are command-line arguments; the single-model scripts do not automatically read YAML configuration files.

The standard TLH implementation performs block-mean calculation and chaotic processing on CPU. Passing `--device cuda` changes model loading and placement, but does not move this TLH processing to CUDA.

## Batch Experiments and Plots

When the optional batch and plotting scripts are included, they support experiments across the eight models and multiple tampering rates.

**Configure checkpoint paths first:** `run_tamper_batch.py` and `run_tamper_baseline_summary.py` contain `MODEL_SPECS` mappings with paths from the original local workspace (`D:\python\...`). Update both mappings before running them on another machine.

Generate tampered models and proposed TLH results:

```bash
python scripts/run_tamper_batch.py --models all --rates 10,20,30,40,50
```

Then generate the no-chaos baseline and combined comparison tables:

```bash
python scripts/run_tamper_baseline_summary.py --models all --rates 10,20,30,40,50
```

The baseline summary requires the checkpoints, logs, and proposed results produced by the first command. Use matching model and rate selections for both commands.

Optional plotting:

```bash
python scripts/plot_tamper_rt.py --input artifacts/tables/tamper_rt_summary.csv
```

Key outputs:

| Path | Contents |
| --- | --- |
| `artifacts/hashes/tlh/` | Proposed TLH JSON and optional text files |
| `artifacts/hashes/tlh_baseline/` | No-chaos baseline TLH files |
| `artifacts/models/tampered/` | Generated tampered checkpoints |
| `artifacts/logs/` | Ground-truth tamper logs, detected differences, and Rt summaries |
| `artifacts/tables/tamper_rt_summary.csv` | Proposed TLH batch results |
| `artifacts/tables/tamper_rt_proposed_vs_baseline_long.csv` | Combined proposed and baseline results |
| `artifacts/tables/tamper_rt_table3_resnets.csv` | ResNet comparison table |
| `artifacts/tables/tamper_rt_table4_other_models.csv` | Other-model comparison table |
| `artifacts/figures/tamper_rt/` | Generated plots |

## Code Organization

| Path | Purpose |
| --- | --- |
| `src/hashing/tlh.py` | Parameter partitioning, block means, chaotic processing, and baseline TLH |
| `src/hashing/encoding.py` | TLH decimal encoding |
| `src/attacks/tamper.py` | Tampering operations and ground-truth block logs |
| `src/evaluation/tamper_accuracy.py` | TLH comparison and localization metrics |
| `src/models/` | Supported model architectures and model factory |
| `src/utils/` | Checkpoint loading and file I/O |
| `src/visualization/` | Optional Rt plotting helpers |
| `scripts/` | Command-line entry points |

PIH functionality is under preparation. Reserved PIH directories may contain `.gitkeep` files so Git can track them before the implementation is added.
