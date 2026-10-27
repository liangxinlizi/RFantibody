# RFantibody
### Structure-Based _de novo_ Antibody Design

![](github_image.png)

# Description
RFantibody is a pipeline for the structure-based design of _de novo_ antibodies and nanobodies. RFantibody consists of three separate methods:
- Protein backbone design with an antibody-finetuned version of [RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8)
- Protein sequence design with [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187)
- _In silico_ filtering of designs using an antibody-finetuned version of [RoseTTAFold2](https://www.biorxiv.org/content/10.1101/2023.05.24.542179v1)

The RFantibody pipeline is described in detail in [this preprint](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1)

# Table of Contents
- [Requirements](#requirements)
- [Downloading Weights](#3-download-model-weights)
- [Installation](#installation)
  - [Local Installation (Without Docker)](#local-installation-without-docker)
  - [Ascend NPU Installation](#ascend-npu-installation)
  - [CUDA Installation](#cuda-installation)
  - [Docker Installation (Alternative)](#docker-installation-alternative)
  - [Apptainer Installation (HPC)](#apptainer-installation-hpc)
- [Command Line Interface](#command-line-interface)
  - [Inference Commands](#inference-commands)
  - [Quiver Utilities](#quiver-utilities)
  - [Full Pipeline Example](#full-pipeline-example)
- [Usage](#usage)
  - [HLT File Format](#hlt-file-format)
  - [Input Preparation](#input-preparation)
  - [RFdiffusion](#rfdiffusion)
  - [ProteinMPNN](#proteinmpnn)
  - [RF2](#rf2)
- [Practical Considerations for Antibody Design](#practical-considerations-for-antibody-design)
  - [Selecting a Target Site](#selecting-a-target-site)
  - [Nanobody Docks](#nanobody-docks)
  - [Truncating your Target Protein](#truncating-your-target-protein)
  - [Picking Hotspots](#picking-hotspots)
  - [Antibody Design Scale](#antibody-design-scale)
  - [Choosing CDR Lengths](#choosing-cdr-lengths)
  - [Filtering Strategies](#filtering-strategies)
- [Quiver Files](#quiver-files)
- [Conclusion](#conclusion)


# Requirements

### Accelerator Support

RFantibody runs on an NVIDIA GPU with CUDA support or on a Huawei Ascend NPU
with the vendor `torch` + `torch_npu` + CANN stack. On an NVIDIA host you can
check whether a compatible GPU is available by running:
```
nvidia-smi
```
If this command runs successfully then RFantibody will be able to run on that
GPU. For the Ascend NPU path see
[Ascend NPU Installation](#ascend-npu-installation). Inference on CPU only is
not supported.

### System Requirements

- **Accelerator**: NVIDIA GPU with CUDA 11.8+ support, **or** Huawei Ascend NPU
  with the vendor PyTorch stack (`torch`, `torch_npu`, CANN)
- **OS**: Linux (Ubuntu 22.04 recommended)


# Installation

## Local Installation (Without Docker)

This is the recommended installation method for most users. It installs RFantibody directly on your system.

### 1. Install uv (Package Manager)

RFantibody uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management. uv will automatically download the required Python version (3.10) if needed.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
After installation, restart your terminal or run:
```bash
source ~/.bashrc  # or ~/.zshrc if using zsh
```

### 2. Clone the Repository

```bash
git clone -b feat/ascend-npu-single-goal-1022 https://github.com/liangxinlizi/RFantibody.git
cd RFantibody
```

### 3. Download Model Weights

```bash
bash include/download_weights.sh
```

### 4. Set Up the Python Environment

Choose the setup for your accelerator:

- [Ascend NPU Installation](#ascend-npu-installation) creates `.venv/` and
  inherits the installed vendor PyTorch stack.
- [CUDA installation](#cuda-installation) creates `.venv-cuda/` and installs
  the CUDA dependencies.

### 5. Activate the Environment

For the Ascend environment:

```bash
source .venv/bin/activate
rfdiffusion --help
proteinmpnn --help
rf2 --help
```

For CUDA, activate `.venv-cuda/bin/activate` instead.

---

## Ascend NPU Installation

RFantibody can also run on Huawei Ascend NPUs through the vendor PyTorch stack
(`torch` + `torch_npu` + CANN).  The vendor stack is treated as read-only: the
project environment inherits it instead of reinstalling PyTorch.

### 1. Create the environment

```bash
# Load the CANN environment (provides ASCEND_HOME_PATH and vendor PYTHONPATH)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Python 3.10 environment that inherits the vendor torch + torch_npu stack
python3.10 -m venv --system-site-packages .venv
.venv/bin/python -m pip install uv
.venv/bin/python -m pip install -e ".[test]"
```

`[test]` installs the test tooling and the CPU DGL build that is used as the
independent reference for graph-operator parity.  `.[npu]` is intentionally
empty: `torch`, `torch_npu` and CANN come from the system installation.  Do not
run `uv sync` in this environment, because it would install a PyPI torch wheel
on top of the vendor stack.

### 2. Build the native AscendC kernels

```bash
bash csrc/ascend_graph/build.sh
```

This compiles the native AscendC graph operators and the tied-attention kernel
against the installed CANN toolkit through the official `ascendc.cmake` route
(`ascendc_library`), and stages `src/rfantibody/ascend_ops/libascend_graph.so`
together with the Python binding used by `rfantibody.ascend_ops`.  The driver
uses `ASCEND_HOME_PATH` (default `/usr/local/Ascend/ascend-toolkit/latest`) and
`.venv/bin/python` unless `PYTHON` is set, and it is idempotent: re-run it after
changing anything under `csrc/ascend_graph/`.  The kernel parity and route tests
under `test/graph_parity/` rebuild the artifact through the same driver when it
is missing or older than its sources.

### 3. Download the model weights

Download the checkpoints into the repository's `weights/` directory:

```bash
bash include/download_weights.sh
export RFANTIBODY_WEIGHTS="$PWD/weights"
```

The pipeline uses `RFdiffusion_Ab.pt`, `ProteinMPNN_v48_noise_0.2.pt`, and
`RF2_ab.pt`. The RF2 JSON tests also use
`RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt`. The download
script obtains all four checkpoints; the files are excluded from Git.

The repository uses `weights/` by default. To select another checkpoint
directory, set `RFANTIBODY_WEIGHTS` or pass `--weights-dir` to `e2e_run.py`.
Individual inference commands accept `--weights` for a checkpoint file.

The test suite selects the versioned `H100_references` set on a genuine Ascend
NPU (`test/util/device_selection.py`) and keeps the original comparison
thresholds; on an A4000/H100 it keeps selecting the matching CUDA reference set.

### 4. Run the tests and the end-to-end pipeline

```bash
export UV_PROJECT="$PWD"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_NO_SYNC=1
.venv/bin/python -m pytest test -ra
.venv/bin/python e2e_run.py            # RFdiffusion -> ProteinMPNN -> RF2 on NPU
```

`e2e_run.py` runs the acceptance chain (RSV site 3 target, `hu-4D5-8_Fv`
framework, the documented hotspots/loops, `diffuser.T=50`, `final_step=1`,
deterministic, then ProteinMPNN `seqs_per_struct=1`/`temperature=0.2`/
`augment_eps=0`, then RF2 with 10 recycles).  It aborts instead of falling back
to CPU and exits 0 only when every stage passes.  The checkpoint directory is
supplied with `--weights-dir`:

```bash
.venv/bin/python e2e_run.py --weights-dir "$PWD/weights"
```

Without the flag it uses `RFANTIBODY_WEIGHTS` or the repository `weights/` directory.

The repository scripts and the Quiver tests invoke `uv run <command>` from
arbitrary working directories.  To make those invocations use the installed
project environment (rather than creating a new one), export:

```bash
export UV_PROJECT=$PWD
export UV_NO_SYNC=1
```

### 5. Reproducible random streams on the NPU

`rfantibody.rng` provides CUDA-compatible Philox streams for the NPU inference
path. It maintains random state per process and device and supports the
normal, exponential, multinomial, and dropout helpers used by the models.
Random variates and masks are generated on CPU and placed on the requested
device. CPU and CUDA inputs use the corresponding PyTorch operations.

The stream layout is configured by `CUDA_RNG_PROFILE` in
`src/rfantibody/rng.py`. Set `RFANTIBODY_CUDA_RNG_PROFILE` to override its
multiprocessor count, threads per multiprocessor, and block size. The supported
distributions, dtype contracts, state transitions, and reference inputs are
covered by `test/rng_parity/`.

### 6. CUDA-order contraction of the RFdiffusion tied attention

RFdiffusion's tied-attention contraction is selected by
`rfantibody.attention_backend`. On NPU, `ascend_graph::tied_attention_logits`
uses an ordered float32 accumulation implemented in AscendC. On CPU and CUDA,
the backend calls the `opt_einsum` contraction.

Build the native operator with `bash csrc/ascend_graph/build.sh`. Its parity,
dispatch, and execution-route tests are in
`test/graph_parity/test_tied_attention_parity.py`; the reference inputs and
expected outputs are bundled in `test/graph_parity/cuda_ordered_holdouts/`.

### 7. CUDA-compatible `cdist` geometry

`rfantibody.rfdiffusion.cdist_compat.cdist` supplies the float32 NPU distance
calculations used by RFdiffusion. For large point sets it implements the
padded-matrix evaluation order with compensated arithmetic on the input
device. It preserves the direct-distance route when both point counts are at
most 25. CPU and CUDA inputs delegate to `torch.cdist`.

The wrapper is used by the template and Cb distance calculations in
`Embeddings.py`, the distance calculations in `Track_module.py`, and the
nearest-neighbour graph in `util_module.py`. The tests in
`test/rfdiffusion/test_cdist_compat.py` cover reference values, shapes, and
broadcasting using the inputs in
`test/rfdiffusion/reference_outputs/cdist_cuda_holdouts/`.

## CUDA Installation

The original CUDA path is preserved in `requirements-cuda.txt`
(torch 2.3 + cu118 and the matching cu118 DGL wheel):

```bash
python3.10 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --upgrade pip
.venv-cuda/bin/python -m pip install -r requirements-cuda.txt
.venv-cuda/bin/python -m pip install -e .
.venv-cuda/bin/python -m pip install uv
```

`requirements-cuda.txt` pins the accelerator packages only, so the editable
install in the third step is what installs the common model dependencies and the
console scripts from `pyproject.toml`; it keeps the cu118 torch and DGL wheels
that step two installed, because `torch>=2.0,<2.6` is already satisfied. Do not
install the `[test]` extra on a CUDA host: it pins the CPU DGL build that is the
independent reference on the Ascend path (`dgl==1.1.3`) and would replace the
cu118 DGL wheel. Install the test tooling on its own instead:

```bash
.venv-cuda/bin/python -m pip install pytest==9.1.1 "isort>=5.13.2"
```

Every console script (`rfdiffusion`, `proteinmpnn`, `rf2`, `qvls`, ...) is then
available under `.venv-cuda/bin/`.

---

## Docker Installation (Alternative)

Docker provides a containerized environment that isolates RFantibody from your host system. This can be useful for:
- Simplified dependency management
- Reproducible environments across different machines
- Avoiding conflicts with existing system packages

### Prerequisites

Install Docker from [here](https://docs.docker.com/engine/install/). If you are running RFantibody on cloud compute, Docker may already be installed. Check by running:
```
which docker
```

You will need to add yourself to the docker group to run containers without sudo:
```
sudo usermod -aG docker $USER
```
After running this command, restart your terminal session for this change to take effect.

### Build the Docker Image

Navigate to the directory where RFantibody is downloaded, then build the Docker image:
```
docker build -t rfantibody .
```

### Start the Docker Container

Run the following command to start the container:
```
docker run --name rfantibody --gpus all -v .:/home --memory 10g -it rfantibody
```
This will put you into the RFantibody container at the `/home` directory which mirrors the directory that you ran the command from.

### Set Up the Python Environment (Inside Container)

From inside the RFantibody container, navigate to the project root and run:
```bash
cd /home
uv sync
```

---

## Apptainer Installation (HPC)

[Apptainer](https://apptainer.org/) (formerly Singularity) is a container platform designed for HPC environments. Unlike Docker, it runs without root privileges and integrates with job schedulers like SLURM.

### Prerequisites

Install Apptainer on Ubuntu:
```bash
sudo apt update
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

### Build the Apptainer Image

Navigate to the RFantibody directory and build the image:
```bash
cd RFantibody
sudo apptainer build rfantibody.sif rfantibody.def
```

This creates a self-contained `rfantibody.sif` file (~8GB) with all dependencies, model weights, and the Python environment pre-installed.

### Running Commands

Always use the `--nv` flag (for GPU support) and `--writable-tmpfs` flag (for temporary file writes):

```bash
# Get help for any command
apptainer exec --nv --writable-tmpfs rfantibody.sif rfdiffusion --help
apptainer exec --nv --writable-tmpfs rfantibody.sif proteinmpnn --help
apptainer exec --nv --writable-tmpfs rfantibody.sif rf2 --help
```

### Bind Mounting Data

Use `-B` to mount directories from your host system:

```bash
# Mount a data directory
apptainer exec --nv --writable-tmpfs \
    -B /path/to/data:/data \
    rfantibody.sif rfdiffusion \
    -t /data/target.pdb \
    -f /data/framework.pdb \
    -o /data/output \
    -n 10
```

### Full Pipeline Example (Apptainer)

```bash
# Set up bind mount
DATA_DIR=scripts/examples
APPTAINER_OPTS="--nv --writable-tmpfs -B $DATA_DIR:/data"

# 1. Design backbones with RFdiffusion
apptainer exec $APPTAINER_OPTS rfantibody.sif rfdiffusion \
    -t /data/example_inputs/flu_HA.pdb \
    -f /data/example_inputs/h-NbBCII10.pdb \
    -q /data/example_outputs/1_app_rfdiffusion.qv \
    -n 2 \
    -l "H1:7,H2:6,H3:5-13" \
    -h "B146,B170,B177"

# 2. Design sequences with ProteinMPNN
apptainer exec $APPTAINER_OPTS rfantibody.sif proteinmpnn \
    -q /data/example_outputs/1_app_rfdiffusion.qv \
    --output-quiver /data/example_outputs/2_app_proteinmpnn.qv \
    -n 4 \
    -t 0.2

# 3. Predict structures with RF2
apptainer exec $APPTAINER_OPTS rfantibody.sif rf2 \
    -q /data/example_outputs/2_app_proteinmpnn.qv \
    --output-quiver /data/example_outputs/3_app_rf2.qv \
    -r 10

# 4. Extract scores to scripts/examples/example_outputs/3_app_rf2.sc
apptainer exec $APPTAINER_OPTS rfantibody.sif qvscorefile /data/example_outputs/3_app_rf2.qv
```

### Interactive Shell

To enter an interactive shell inside the container:
```bash
apptainer shell --nv --writable-tmpfs rfantibody.sif
```

Type `exit` or press `Ctrl+D` to exit.

### SLURM Integration

Example SLURM batch script:
```bash
#!/bin/bash
#SBATCH --job-name=rfantibody
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

apptainer exec --nv --writable-tmpfs \
    -B /scratch/$USER:/data \
    /path/to/rfantibody.sif rfdiffusion \
    -t /data/target.pdb \
    -f /data/framework.pdb \
    -o /data/designs \
    -n 100
```

# Command Line Interface

RFantibody provides a set of command-line tools for running the design pipeline and working with Quiver files. After setting up the environment, these commands are available directly in your terminal.

## Inference Commands

### RFdiffusion (Backbone Design)

```bash
rfdiffusion -t antigen.pdb -f framework.pdb -o designs/ab -n 10
```

Key options:
- `-t, --target`: Target PDB file (antigen)
- `-f, --framework`: Framework PDB file (antibody scaffold)
- `-o, --output`: Output prefix for designs
- `-q, --output-quiver`: Output to Quiver file instead of PDBs
- `-n, --num-designs`: Number of designs to generate
- `-l, --design-loops`: Loop lengths, e.g., `"H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11"`
- `-h, --hotspots`: Hotspot residues, e.g., `"A305,A456"`
- `--deterministic`: Enable reproducible results

Example with all options:
```bash
rfdiffusion \
    -t target.pdb \
    -f framework.pdb \
    -q designs.qv \
    -n 100 \
    -l "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11" \
    -h "T305,T456" \
    --deterministic
```

### ProteinMPNN (Sequence Design)

```bash
proteinmpnn -i structures/ -o sequences/ -n 5
```

Key options:
- `-i, --input-dir`: Input directory of PDB files
- `-q, --input-quiver`: Input Quiver file
- `-o, --output-dir`: Output directory for PDB files
- `--output-quiver`: Output Quiver file
- `-l, --loops`: Loops to design (default: `H1,H2,H3,L1,L2,L3`)
- `-n, --seqs-per-struct`: Sequences per structure
- `-t, --temperature`: Sampling temperature (default: 0.1)
- `--deterministic`: Enable reproducible results

Example with Quiver files:
```bash
proteinmpnn -q backbones.qv --output-quiver sequences.qv -n 5 -t 0.2
```

### RF2 (Structure Prediction)

```bash
rf2 -i structures/ -o predictions/
```

Key options:
- `-p, --input-pdb`: Single input PDB file
- `-i, --input-dir`: Input directory of PDB files
- `-q, --input-quiver`: Input Quiver file
- `-o, --output-dir`: Output directory for PDB files
- `--output-quiver`: Output Quiver file
- `-r, --num-recycles`: Recycling iterations (default: 10)
- `-s, --seed`: Random seed for reproducibility
- `--hotspot-show-prop`: Proportion of hotspot residues to show to model (default: 0.1)

Example with Quiver files:
```bash
rf2 -q sequences.qv --output-quiver predictions.qv -r 10
```

## Quiver Utilities

Commands for working with Quiver files (protein design databases):

| Command | Description |
|---------|-------------|
| `qvls` | List all tags in a Quiver file |
| `qvextract` | Extract all PDB files from a Quiver |
| `qvextractspecific` | Extract specific PDBs by tag name |
| `qvscorefile` | Extract scores to a TSV file |
| `qvsplit` | Split into multiple files |
| `qvslice` | Extract specific tags to new Quiver |
| `qvrename` | Rename tags in a Quiver file |
| `qvfrompdbs` | Create a Quiver from PDB files |

All commands support `--help` for detailed usage information.

## Full Pipeline Example

We provide a complete pipeline script at `scripts/examples/nanobody_full_pipeline.sh` that runs all three steps of the RFantibody workflow. Here's a walkthrough:

### Configuration

The script defines editable parameters at the top:

```bash
# Input files
TARGET_PDB="scripts/examples/example_inputs/flu_HA.pdb"       # Target antigen
FRAMEWORK_PDB="scripts/examples/example_inputs/h-NbBCII10.pdb" # Nanobody framework

# Output directory
OUTPUT_DIR="scripts/examples/example_outputs/nb_ha_pipeline"

# RFdiffusion parameters
NUM_DESIGNS=1000                        # Number of backbone designs
DESIGN_LOOPS="H1:7,H2:6,H3:5-13"        # CDR loop lengths
HOTSPOTS="B146,B170,B177"               # Target residues to focus binding

# ProteinMPNN parameters
NUM_SEQS=4                              # Sequences per backbone
SAMPLING_TEMP=0.2                       # Sampling temperature

# RF2 parameters
NUM_RECYCLES=10                         # Recycling iterations
```

### Step 1: RFdiffusion (Backbone Design)

```bash
uv run rfdiffusion \
    --target "$TARGET_PDB" \
    --framework "$FRAMEWORK_PDB" \
    --output-quiver "$OUTPUT_DIR/1_rfdiffusion.qv" \
    --num-designs "$NUM_DESIGNS" \
    --design-loops "$DESIGN_LOOPS" \
    --hotspots "$HOTSPOTS"
```

This generates nanobody backbone structures docked to the target, with CDR loops designed to contact the specified hotspot residues.

### Step 2: ProteinMPNN (Sequence Design)

```bash
uv run proteinmpnn \
    --input-quiver "$OUTPUT_DIR/1_rfdiffusion.qv" \
    --output-quiver "$OUTPUT_DIR/2_proteinmpnn.qv" \
    --seqs-per-struct "$NUM_SEQS" \
    --temperature "$SAMPLING_TEMP"
```

Takes the backbone designs and generates amino acid sequences for the CDR loops. Multiple sequences per backbone increases diversity.

### Step 3: RF2 (Structure Prediction)

```bash
uv run rf2 \
    --input-quiver "$OUTPUT_DIR/2_proteinmpnn.qv" \
    --output-quiver "$OUTPUT_DIR/3_rf2.qv" \
    --num-recycles "$NUM_RECYCLES"
```

Predicts the final structures and provides confidence scores (pLDDT, PAE) for filtering.

### Working with Results

After the pipeline completes, use Quiver utilities to analyze results:

```bash
# List all designs
qvls $OUTPUT_DIR/3_rf2.qv

# Extract information on how well RFdiffusion targeted the hotspots for each design
qvscorefile $OUTPUT_DIR/1_rfdiffusion.qv

# Extract RF2 scores to a tab-separated scorefile (3_rf2.sc)
qvscorefile $OUTPUT_DIR/3_rf2.qv

# Extract all PDBs for visual inspection
qvextract $OUTPUT_DIR/3_rf2.qv -o final_designs/
```

### Running the Pipeline

```bash
bash scripts/examples/nanobody_full_pipeline.sh
```

The script outputs three Quiver files representing each stage, allowing you to inspect intermediate results or restart from any step.

# Usage

> **Note:** The examples below assume you have the RFantibody environment active. Either activate it with `source .venv/bin/activate`, or prefix each command with `uv run` (e.g., `uv run rfdiffusion ...`).

## HLT File Format
We must pass structures between the different steps of the RFantibody pipeline. Each step of the pipeline must know:
- The antibody-target complex structure we are currently designing for
- Which chain is the Heavy chain, Light chain, and Target chain
- Which residues are in which of the CDR loops

To enable the passing of this information between steps of the pipeline, we define a file format that we call an HLT file. An HLT file is simply a .pdb file but with the following modifications:
- The Heavy chain is denoted as chain id 'H'
- The Light chain is denoted as chain id 'L'
- The Target chain(s) are denoted as chain id 'T' (even if there are multiple target chains)
- The order of the chains in the file is Heavy then Light then Target
- At the end of the file are PDB Remarks indicating the 1-indexed absolute (not per-chain) residue index of each of the CDR loops. For example:
  ```
  REMARK PDBinfo-LABEL:   32 H1
  REMARK PDBinfo-LABEL:   52 H2
  ```

## Input Preparation

The antibody-finetuned version of RFdiffusion in RFantibody requires an HLT-remarked framework structure as input. We provide a script to perform this conversion:
```bash
python scripts/util/chothia_to_HLT.py -inpdb mychothia.pdb -outpdb myHLT.pdb
```

This script expects a Chothia annotated .pdb file. A great source for these files is [SabDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab), which provides Chothia annotated structures of all antibodies and nanobodies in the PDB and is updated every few months.

We provide the HLT-formatted antibody and nanobody frameworks that were used in the design campaigns from the RFantibody preprint here:
```
Nanobody Framework: scripts/examples/example_inputs/h-NbBCII10.pdb
ScFv Framework: scripts/examples/example_inputs/hu-4D5-8_Fv.pdb
```

## RFdiffusion

The first step in RFantibody is to generate antibody-target docks using an antibody-finetuned version of RFdiffusion. Here is an example command that will run RFdiffusion:
```bash
rfdiffusion \
    -t scripts/examples/example_inputs/rsv_site3.pdb \
    -f scripts/examples/example_inputs/hu-4D5-8_Fv.pdb \
    -o scripts/examples/example_outputs/ab_des \
    -n 20 \
    -l "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11" \
    -h "T305,T456"
```

Let's go through this command in more detail to understand what these options are doing:
- `-t, --target`: A path to the target structure that we wish to design antibodies against. This is commonly a cropped target structure to reduce the computational expense of running the pipeline. Cropping strategies are explained in more depth [here](#truncating-your-target-protein).
- `-f, --framework`: A path to the HLT-formatted antibody framework that we wish to use for our design. RFdiffusion will only design the structure and sequence of regions of the framework which are annotated as loops, this allows us to design the dock and loops of already optimized frameworks.
- `-o, --output`: The prefix of the .pdb file outputs that we will generate.
- `-n, --num-designs`: The number of designs we should generate.
- `-l, --design-loops`: A mapping of each CDR loop to a range of allowed loop lengths. The length of each loop is sampled uniformly from this range and is sampled independently of the lengths sampled for other loops. If a CDR loop exists in the framework but is not specified, this CDR loop will have its sequence and structure fixed during design. If a CDR loop is included but no range of lengths is provided (e.g., `H1:7`), this CDR loop will have its sequence and structure designed but only with the specified length.
- `-h, --hotspots`: A list of hotspot residues that define our epitope. We discuss selecting hotspots in more detail [here](#selecting-a-target-site).

For Quiver file output, use `-q` instead of `-o`:
```bash
rfdiffusion \
    -t target.pdb \
    -f framework.pdb \
    -q designs.qv \
    -n 100 \
    -l "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11" \
    -h "T305,T456"
```

Run `rfdiffusion --help` to see all available options.

## ProteinMPNN

The second step in RFantibody is to take the docks generated by RFdiffusion and assign sequences to the CDR loops. We do this using the base version of ProteinMPNN, ie. not an antibody-finetuned model. For convenience, we package the necessary ProteinMPNN scripts in this repo and provide a wrapper that enables the design of just the CDR loops using ProteinMPNN.

At its simplest, ProteinMPNN may be run on a directory of HLT-formatted .pdb files using the following command:
```bash
proteinmpnn -i /path/to/inputdir -o /path/to/outputdir
```

This will design all CDR loops and will provide one sequence per input structure. To generate multiple sequences per structure or use Quiver files:
```bash
# Generate 5 sequences per structure with Quiver I/O
proteinmpnn -q backbones.qv --output-quiver sequences.qv -n 5

# Specify which loops to design and adjust temperature
proteinmpnn -i structures/ -o sequences/ -l "H1,H2,H3" -t 0.2
```

Run `proteinmpnn --help` to see all available options.

## RF2

The final step of the RFantibody pipeline is to use our antibody-finetuned RF2 to predict the structure of the sequences we just designed. We then assess whether RF2 is confident that the sequence will bind as we designed.

At its simplest, RF2 may be run on a directory of HLT-formatted .pdb files using the following command:
```bash
rf2 -i /path/to/inputdir -o /path/to/outputdir
```

By default this will run with 10 recycling iterations and with 10% of hotspots provided to the model. We don't yet know what combination of these hyperparameters will be most predictive of design success but it should be possible to tune these values once we have data on more antibody and nanobody campaigns.

For Quiver file I/O or to adjust recycling iterations:
```bash
# Use Quiver files with 10 recycles
rf2 -q sequences.qv --output-quiver predictions.qv -r 10

# Run on a single PDB file
rf2 -p design.pdb -o predictions/
```

### JSON Input

RF2 also accepts JSON files containing antibody sequences and target structures. This is useful when you have antibody sequences (without coordinates) and a known target structure. Each entry in the JSON contains heavy/light chain sequences and the target's 3D coordinates:

```bash
rf2 -j targets.json -o predictions/
```

The JSON file should be a dictionary keyed by sample ID, where each entry has:
- `Hseq`: Heavy chain amino acid sequence (one-letter codes)
- `Lseq`: Light chain amino acid sequence (one-letter codes)
- `hotspots`: Boolean array marking target interface residues (length = target length)
- `T`: Target structure dict with `seq` (integer-encoded), `xyz` (Lx27x3 coordinates), `mask` (Lx27 atom mask), `idx`, `pdb_idx`, and `cdr_bool`

### TCR-MHC Prediction

For TCR structure prediction against MHC-peptide targets, use the JSON input format with the TCR-specific weights:

```bash
rf2 -j tcr_targets.json -o predictions/ \
    --weights weights/RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt \
    --hotspot-show-prop 0
```

The TCR weights (`RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt`) are downloaded automatically by `include/download_weights.sh` and use the older BinderNetwork architecture, which is auto-detected at load time.

Run `rf2 --help` to see all available options.

# Practical Considerations for Antibody Design
Designing antibodies is similar to designing _de novo_ binders but is in an earlier stage of development. Here we share advice and learnings on how best to use this pipeline to design antibodies which will work experimentally. We expect some of this advice to change as more antibody design campaigns are performed and best-practices crystallize. Several of these sections are adapted from the analogous section of the RFdiffusion README as these two methods share many similarities and the advice applies to both.

## Selecting a Target Site

Not every site on a target protein is a good candidate for antibody design. For a site to be an attractive candidate for binding it should have >~3 hydrophobic residues for the binder to interact with. Binding to charged polar sites is still quite hard. Binding to sites with glycans close to them is also hard since they often become ordered upon binding and you will take an energetic hit for that. Binding to unstructured loops has historically been hard but [this paper](https://www.nature.com/articles/s41586-023-06953-1) outlines a strategy to use RFdiffusion to bind unstructured peptides which share much in common unstructured loops, using this strategy should work with antibodies but depending on the flexibility of the loop, you will pay an energetic price for ordering the loop during binding.

## Nanobody Docks

When you begin looking at your nanobody outputs, you may notice that many are binding in a side-on dock. This is not a bug and is a result of the model being trained on natural nanobody docks which often bind in this side-on docking style and make some framework-mediated contacts. You may be able to tune your hotspots and CDR lengths to get a more antibody-like dock, but we recommend that if you desire an antibody-like dock, then you should design with an antibody framework.

## Truncating your Target Protein

RFdiffusion and RF2 scale in runtime as O(N^2) where N is the number of residues in your system. As such, it is a very good idea to truncate large targets so that your computations are not unnecessarily expensive. All steps in the RFantibody pipeline are designed to allow for a truncated target. Truncating a target is an art. For some targets, such as multidomain extracellular membranes, a natural truncation point is where two domains are joined by a flexible linker. For other proteins, such as virus spike proteins, this truncation point is less obvious. Generally you want to preserve secondary structure and introduce as few chain breaks as possible. You should also try to leave ~10A of target protein on each side of your intended target site. We recommend using PyMol to truncate your target protein.

## Picking Hotspots

Hotspots are a feature that we integrated into the model to allow for the control of the site on the target which the antibody will interact with. During training, we classify a target residue as a hotspot if it has an average Cβ distance to the closest 5 antibody CDR residues of less than 8 Angstroms. Of all of the hotspots which are identified on the target 0-100% of these hotspots are actually provided to the model and the rest are masked. We find that RFantibody is more sensitive to exactly which hotspots are selected than vanilla RFdiffusion is. Where RFdiffusion tends to generative long helices when given a bad set of hotspots, RFantibody will generally just generate an undocked antibody if a bad set of hotspots is given. It is a very good idea to run a few pilot runs before generating thousands of designs to make sure the number of hotspots you are providing will give results you like.

## Antibody Design Scale

For some of the target campaigns that we report on in our manuscript, we were able to identify VHH binders from a set of 95 designs. In the more general case, however, we expect that design campaigns in the 10k range will be required to identify hits. This is in large part due to the lack of a reliable filtering metric (discussed further in the [Filtering Strategies](#filtering-strategies) section). All data, both positive and negative, is useful for tuning and evaluating filters so if you run a design campaign and wish to share your data with the broader community that would be extremely helpful for moving toward a more reliable filter, higher success rates, and cheaper design campaigns.

## Choosing CDR Lengths

The loop ranges that we used for our design campaigns are provided in the RFdiffusion example files. We determined these ranges by looking at the frequency of naturally occuring lengths for each loop and trying to cover most of the density with our range. We also tried to choose relatively short H3 loops, as we figured these would be easier to design and predict while still giving us enough length to bind effectively. There are some targets where having a long H3 may be useful, for instance when targeting a hydrophobic pocket in a protein. In these cases, the H3 range should be increased beyond what we provide in the examples.

## Filtering Strategies

We recommend the following minimal filtering critieria: <br />
<br />
RF2 pAE < 10 <br />
RMSD (design versus RF2 predicted) < 2&#197; <br />
It may also be helpful to filter by Rosetta ddG < -20 <br />
<br />
The lack of an effective filter is the main limitation of the RFantibody pipeline at the moment. The version of RF2 that we provide may show weak enrichment of binders over non-binders in some cases but more data is needed to make this conclusion convincingly. Newly available structure prediction models such as AF3 present a promising alternative to RF2 and we are in the process of evaluating these models for predictivity on our design campaigns.

# Quiver Files

When running large-scale design campaigns it is often useful to have a single file which holds many designs and the scores associated with those designs. This is gentler on file systems than storing and accessing thousands of individual .pdb files. We offer the ability to use [Quiver files](https://github.com/nrbennet/quiver) in the RFantibody pipeline. These files are simply one large file with the contents of many smaller files inside of them. Each entry has a unique name and can store meta_data about the entry.

RFantibody provides command line tools for working with Quiver files. These are composable (pipe-able) commands inspired by Brian Coventry's [silent_tools](https://github.com/bcov77/silent_tools) project. Use `--help` with any command for detailed options.

```bash
# make a quiver file
qvfrompdbs *.pdb > my.qv

# ask what's in a quiver file
qvls my.qv

# ask how many things are in a quiver file
qvls my.qv | wc -l

# extract all pdbs from a quiver file
qvextract my.qv

# extract to a specific directory
qvextract my.qv -o output_dir/

# extract the first 10 pdbs from a quiver file
qvls my.qv | head -n 10 | qvextractspecific my.qv

# extract a random 10 pdbs from a quiver file
qvls my.qv | shuf | head -n 10 | qvextractspecific my.qv

# extract a specific pdb from a quiver file
qvextractspecific my.qv name_of_pdb_0001

# produce a scorefile from a quiver file
qvscorefile my.qv > scores.tsv

# combine qv files
cat 1.qv 2.qv 3.qv > my.qv

# rename tags in a quiver file
qvls my.qv | sed 's/$/_v2/' | qvrename my.qv > renamed.qv

# slice specific tags into a new quiver file
qvls | shuf | head -n 10 | qvslice > subset.qv

# split a quiver file into groups of 100
qvsplit my.qv 100 -o split_dir/
```

## Reading and Writing Quiver Files
All steps of RFantibody allow for the use of Quiver files. The syntax is summarized here:

RFdiffusion takes only a .pdb file target and framework as input. To output the designed backbones at quiver files append this argument to your input command:
```
inference.quiver=/path/to/myoutput.qv
```

For ProteinMPNN, to input and output a Quiver file, use the following two arguments:
```
-inquiver /path/to/myinput.qv -outquiver /path/to/myoutput.qv
```

RFantibody takes the following two configs to work with Quiver file input and output
```
input.quiver=/path/to/myinput.qv output.quiver=/path/to/myoutput.qv
```

# Conclusion
We are really excited to release RFantibody open-source! We can't wait to see what kinds of designs the broader community comes up with. We have worked hard to make this codebase as easy to setup and run as possible but please open a GitHub issue if you run into any problems.

\- Nate, Joe, and the RFantibody Team

---

RFantibody builds directly off of the architecture and weights of several methods which we acknowledge here. We thank Minkyung Baek and Frank DiMaio for developing RoseTTAFold and RoseTTAFold2 which the original RFdiffusion and our antibody-fine tuned RoseTTAFold2 model are based off of. We thank Justas Dauparas for developing ProteinMPNN which we provide an antibody-specific wrapper for in this repo. As the antibody-finetuned RFdiffusion we provide here is directly based off of the original version of RFdiffusion, we also thank David Juergens, Brian Trippe, and Jason Yim who co-developed the original RFdiffusion with us. RFantibody is released under an MIT License (see LICENSE file). It is free for both non-profit and for-profit use.
