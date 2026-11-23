# LLM Hallucination Detection using Supervised Machine Learning

Implementation code for the machine learning project based on the [RAGTruth dataset](https://github.com/ParticleMedia/RAGTruth).

## Project overview

The notebook compares Logistic Regression and Linear SVM on two TF-IDF text representations: generated responses alone, and source text combined with generated responses. It also includes a majority-class baseline, source-grouped cross-validation, a held-out test evaluation, error analysis, and a **separate exploratory feature-engineering experiment** using source–response overlap and numeric features.

## Contents

- `LLM_Hallucination_Detection.ipynb` — complete analysis and saved results
- `requirements.txt` — Python package dependencies
- `.gitignore` — excludes downloaded datasets, trained model binaries and temporary files

## How to run

1. Use Python 3.13 or another compatible Python 3 environment.
2. In this folder, create and activate a virtual environment, or select an existing suitable environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start Jupyter from the project folder:

   ```bash
   python -m notebook
   ```

5. Open `LLM_Hallucination_Detection.ipynb` and run cells **from top to bottom**.

The first data-loading section downloads `response.jsonl` and `source_info.jsonl` into `data/` if they are absent. An internet connection is needed for the initial download. Tables, figures, and the trained final model are saved in `outputs/`. The downloaded dataset and trained model binary are not needed in the GitHub repository to reproduce the analysis.

## Evaluation design

- Uses the official RAGTruth training/test partition.
- Makes a source-grouped training/validation split using the official training data.
- Tunes classifier regularization using 3-fold stratified, source-grouped cross-validation.
- Fits TF-IDF inside each training fold through a scikit-learn pipeline.
- Selects one model based on hallucination-class validation F1, refits it on the official training partition, and evaluates it on the held-out official test set.
- Reports the later feature-engineering comparison **on validation only**; it is exploratory and is not a new held-out test result.

## Results

The saved notebook outputs include the final response-only Linear SVM test accuracy (approximately 0.739) and hallucination F1 (approximately 0.636). The separate feature-engineering experiment raises Logistic Regression's validation F1 from approximately 0.667 to 0.728. These numbers come from different evaluation subsets and should not be compared as though both were final test scores.

## Source

Niu et al. (2024), *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models*. https://aclanthology.org/2024.acl-long.585/
