# ReScaL

This repository contains the data and code for the following paper: 
> Augmenting Scaling Laws with Residual Learning for LLM Pretraining Performance Prediction

## Abstract
> The computational cost of pretraining Large Language Models (LLMs) at modern scales makes it essential to predict pretraining outcomes (validation loss on held-out data) before committing to expensive full-scale pretraining. Scaling laws enable extrapolation from small-scale pilots but miss feature-dependent variations, while purely data-driven methods require substantial training data and extrapolate unreliably beyond observed regimes. To address this dual limitation, we propose ReScaL (Residual-augmented Scaling Laws), a hybrid method that partially fits and ensembles multiple scaling laws on minimal pilot runs; learns ML-predicted residual corrections to capture feature-dependent variations that scaling laws miss; and dynamically combines scaling-law extrapolation with residual corrections to leverage the strengths of both approaches. Through experiments on 52 pairs of pretraining datasets and budget sizes, we show that ReScaL is ranked 1st in 42/52 cases (81\%) with the best average rank of 1.21, and attains up to 5.7× lower MAPE than the strongest counterpart. Ablations show that removing each design can degrade accuracy by up to 4.3×, while sensitivity analysis confirms our hyperparameter choice as a robust operating point.


This repository contains the **key codes**, **full data used**, and the **raw experiment results** for the paper.

## Documents

- **data.zip**:

LLM pre-train datasets as specified in the paper.

- **result.zip**:

All the raw experimental data and their final statistical results (including rank, mean, and std) are stored in this directory.

- **Method**:

Model and method implementation code.


- **test**:

scripts to reproduce our experiments.

## Pre-run Setup

Before running any experiments or demos, please complete the following setup steps.

### Step 1: Unzip data and results

Unzip the provided archives into the project root directory:

```bash
unzip data.zip
unzip result.zip
```

After extraction, the directory structure should include:

```
data/
result/
```

---

### Step 2: Install dependencies

Install all required Python dependencies using:

```bash
pip install -r requirements.txt
```


## Using Custom Datasets

ReScaL supports evaluation on **user-provided datasets**. To test the method on a new dataset, please follow the steps below.

### Step 1: Prepare the dataset

1. Create a new subdirectory under the `data/` directory, e.g.,

   ```
   data/my_dataset/
   ```
2. Group your data according to the **scale variables** (N) (model size) and (D) (training data size).
3. Save each group into separate files within this directory.
   Each data point should be associated with a unique ((N, D)) pair, following the same format as existing datasets in `data/`.

> **Note:** The method assumes that (N) and (D) are explicitly available and comparable across all samples.

---

### Step 2: Configure `demo.py`

Open `test/demo.py` and modify the following fields:

1. **Register the dataset path**
   At **line 10**, add the new dataset entry to `system_dir_dict`:

   ```python
   system_dir_dict["my_dataset"] = "data/my_dataset"
   ```

2. **Specify the training sampling ratio**
   At **line 21**, add the dataset to `system_test_arch_dict` and set the sampling ratio (ρ):

   ```python
   system_test_arch_dict["my_dataset"] = [ρ]
   ```

   Here, ρ denotes the proportion of the **least expensive** data (ranked by training cost) used for training.

3. **Declare all ((N, D)) pairs**
   At **line 32**, add all **unique and non-duplicated** ((N, D)) pairs of the dataset to `all_nd_pair`.

4. **Set the result output directory**
   Specify the path where prediction results and evaluation outputs will be saved.

---

### Step 3: Run the demo

After completing the configuration, execute:

```bash
python test/demo.py
```

The script will automatically perform diverse fitting, ensemble prediction, and evaluation on the new dataset, with results saved to the specified output directory.

## Reproducing Experimental Results

This repository provides ready-to-run scripts for reproducing all experimental results reported in the paper.

### Step 1: Run experiment scripts

Each research question (RQ) corresponds to a dedicated script located in the `test/` directory.

* To reproduce **RQ1**, run:

  ```bash
  python test/test_RQ1_sota.py
  ```
* Similarly, other RQs can be reproduced by executing their corresponding scripts (e.g., `test_RQ2_*.py`, `test_RQ3_*.py`).

All experiment scripts will **automatically execute the full evaluation pipeline** and save the raw results (predictions, errors, intermediate metrics) to the predefined result directory.

No additional configuration is required.

---

### Step 2: Statistical analysis and ranking

After all experiments finish, run the following script to aggregate and analyze the results:

```bash
python test/get_stats.py
```

This script will:

* Compute **mean** and **variance** of prediction errors for each method
* Perform **statistical ranking** using **Scott–Knott ESD**
* Output method ranks and summarized statistics used in the paper

---

### Dependency note (Scott–Knott ESD)

The Scott–Knott ESD test relies on an **R language** implementation.

Please install R and the required packages following the instructions provided at:

> [https://github.com/klainfo/ScottKnottESD](https://github.com/klainfo/ScottKnottESD)

Ensure that R is correctly configured and accessible from the command line before running `get_stats.py`.


## Baseline Scaling Law Models

We compare **ReScaL** against a diverse set of representative **baseline scaling law models**, covering compute-optimal, data-constrained, and data-mixture-aware formulations. These baselines represent the major lines of work in scaling law research.

### 1. FARSEER Scaling Law

**Reference:** [https://arxiv.org/abs/2506.10972](https://arxiv.org/abs/2506.10972)



### 2. Chinchilla Scaling Law

**Reference:** [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)



### 3. Kaplan Scaling Law

**Reference:** [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)



### 4. Data-Constrained Scaling Law

**Reference:** [https://arxiv.org/abs/2305.16264](https://arxiv.org/abs/2305.16264)



### 5. Data Mixing Law

**Reference:** [https://arxiv.org/abs/2403.16952](https://arxiv.org/abs/2403.16952)









