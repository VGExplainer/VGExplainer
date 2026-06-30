## Dataset

The old version of Big-Vul is available at:

- [MSR_data_cleaned.zip - Google Drive](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view)

The new version of Big-Vul is available at:

- [Big-Vul updated folder - Google Drive](https://drive.google.com/drive/folders/1VPUGYjrhIEXYOdPjYGdwYrHfvGb4LL7O)

PrimeVul is available at:

- [DLVulDet/PrimeVul: Repository for PrimeVul Vulnerability Detection Dataset](https://github.com/DLVulDet/PrimeVul)

## Requirement

Please check all dependencies in `requirements.txt`.

For LLM-based experiments, Ollama can be downloaded from:

- [Download Ollama on Linux](https://ollama.com/download/linux)

## Preprocess

Please check the `Preprocess` folder.

1. Run `preprocess/get_code_bigvul.py` to extract code from Big-Vul and generate patch lines.  
   Alternatively, you can use our extracted code package in `Big_Vul_updated.zip`.

   Run `preprocess/get_code_primevul.py` to extract code from PrimeVul, and use `get_diff.py` to generate patch lines.

2. Run `preprocess/normalize/normalization.py` to normalize the code.

3. Use Joern to generate PDG graphs. We use Joern `v4.0.408`.  
   Please refer to the Joern project for graph-generation details:

   - [Joern](https://github.com/joernio/joern)

   We provide `preprocess/joern_graph_gen.py` for this step. Required intermediate files include:

   - `.bin`: used for generating dot graphs and line-to-node dictionaries
   - `.dot`: the PDG of the code
   - `.json`: a dictionary used to map node IDs to line IDs

4. Use `get_behavior/get_behaviors.py` and `get_gt.py` to generate ground truth and local behaviors.

5. Run `preprocess/train_w2v.py` to train the Word2Vec model.

6. Run `preprocess/embedding.py` to generate the data required by the vulnerability detection model.

## Train or Test VD Model

You need to modify `train.py` and `test.py`, especially the input and output paths.

## Run VGExplainer

Before running VGExplainer:

- Check `benchmark/data/dataset.py` and `dataset_gen.py` to specify and load the graphs to explain.
- Modify `benchmark/args.py` for the trained model path and explainer arguments.
- Modify `benchmark/models/models.py` for the vulnerability detection models.
- Modify `benchmark/models/explainers.py` to specify the dot path, slice path, and output path.

Then run:

- `benchmark/kernel/pipeline.py`

## Evaluation

First use:

- `rq1-3_evaluation/node2line.py`

to convert nodes into source lines.

Then use:

- `rq1-3_evaluation/evaluation.py`

to compute:

- MSP
- MSR
- MIoU

## LLM-Assisted Vulnerability Analysis

Use:

- `rq4_evaluation/add_num.py` to add line numbers to source code.

Then run:

- `rq4_evaluation/llm_vd.py` for detection **without** explainers
- `rq4_evaluation/llm_vd_enhanced.py` for detection **with** explainers

The prompt templates are included directly in the code.

For evaluation, use:

- `rq4_evaluation/eval_tp.py` to compute VIR, MSP, MSR, and MIoU.
- `rq4_evaluation/eval_fp.py` to compute FPRR.

## Case Study

In addition to CVE-2017-6892 discussed in the paper, we also provide vulnerability explanations for VGExplainer in five real-world scenarios:

- `cases/CVE-2015-8961` : CWE-416
- `cases/CVE-2017-12179` : CWE-190
- `cases/CVE-2018-15863` : CWE-476
- `cases/CVE-2019-15296` : CWE-119
- `cases/CVE-2020-13910` : CWE-125

## Stability & Robustness

Due to space limitations, we refer to prior work for stability and robustness results on the Big-Vul dataset:

[1] Hu, Y., Wang, S., Li, W., Peng, J., Wu, Y., Zou, D., & Jin, H. (2023, July). *Interpreters for GNN-based vulnerability detection: Are we there yet?* In *Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis* (pp. 1407-1419).
