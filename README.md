# Board Game Recommender

Board Game Recommender is a CMPE256 final project focused on building and comparing recommendation approaches for BoardGameGeek (BGG) data.  
The repository contains exploratory analysis plus multiple model experiments, including baseline collaborative filtering, hybrid recommenders, and LLM-augmented embeddings.

## Project Goals

- Explore large-scale BGG metadata and user-rating behavior.
- Build baseline recommendation models for comparison.
- Evaluate improved models such as hybrid and graph-based methods.
- Experiment with LLM-derived item representations to improve recommendations.

## Repository Structure

- `eda.ipynb`: Main exploratory data analysis notebook.
- `baseline_popularity_based.ipynb`: Popularity-based baseline recommender.
- `baseline_svd.ipynb`: SVD-based collaborative filtering baseline.
- `users/manu/v1/`: Final submission folder with the primary Two-Tower hybrid modeling workflow.
- `users/manu/v1/[v1]_llm-embedding.ipynb`: Synthesis of rich-text profiles and 768-dim vector generation.
- `users/manu/v1/[v1]_nnw.ipynb`: Primary neural network training and final evaluation on the test set.
- `users/manu/v2/`: Exploratory folder for feature-sensitivity analysis using structured metadata (designer, year, complexity).
- `users/manu/v2/[v2]_llm-embedding.ipynb`: Enriched text profile generation.
- `users/manu/v2/[v2]_nnw.ipynb`: Model training for metadata-augmented embeddings.
- `users/manu/_drafts/`: Development history and initial data checks.
- `users/rahul/EDA_Rahul.ipynb`: EDA variant by Rahul.
- `users/rahul/Hybrid_recommender_Rahul.ipynb`: Hybrid recommender experiment.
- `users/darshan/lightgcn_recommender.ipynb`: Secondary LightGCN iteration.
- `users/darshan/v1/darshan_lightgcn.ipynb`: Initial lightGCN iteration
- `users/darshan/hyperparameter_search.txt`: Hyperparameter testing

This repository is notebook-driven and does not currently include a packaged Python module or CLI entrypoint.

## Data Overview

The notebooks reference a BGG-derived dataset with:

- Raw game metadata (`raw_data/games.csv`) and related feature tables (`themes`, `mechanics`, `designers`, `artists`, `publishers`, etc.).
- Raw user interactions (`raw_data/user_ratings.csv`) with tens of millions of rating records.
- Processed parquet/csv artifacts for training and evaluation splits in some workflows.

## Dataset Access

Primary dataset source:

- [Board Games Database from BoardGameGeek (Kaggle)](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek)

After downloading, organize files to match the paths used by notebooks (for example, `raw_data/games.csv` and `raw_data/user_ratings.csv` under your project data root).

Most notebooks currently expect files mounted from a shared Google Drive path:

`/content/drive/Shareddrives/CMPE256_FinalProject/board_game_recommendation`

Dataset files are not included in this repository. If you run locally, update the path variables in each notebook to point to your data location.

## Environment Setup

Recommended: Python 3.10+ and Jupyter/Colab.

1. Create and activate a virtual environment.
2. Install dependencies used across notebooks:

```bash
pip install jupyter polars pandas scipy scikit-learn matplotlib seaborn torch pyarrow sentence-transformers lightgbm "numpy<2.0" scikit-surprise
```

Notes:

- `scikit-surprise` currently works most reliably with `numpy<2.0`.
- For faster neural training, use a GPU-enabled PyTorch runtime (for example, Colab GPU).

## Running the Project

### Option 1: Google Colab (matches current notebook paths)

1. Upload/open notebooks in Colab.
2. Mount Google Drive when prompted (many notebooks include `drive.mount('/content/drive')`).
3. Confirm that the base dataset path points to:
   `/content/drive/Shareddrives/CMPE256_FinalProject/board_game_recommendation`
4. Install any notebook-specific extras when prompted in cells (`lightgbm`, `sentence-transformers`, `scikit-surprise`).
5. Run cells top-to-bottom.

### Option 2: Local Jupyter

1. Place raw/processed data in a local directory.
2. Start Jupyter:

```bash
jupyter notebook
```

3. Open each notebook and replace Google Drive paths with your local base path.
4. If a notebook has Colab-specific cells (`from google.colab import drive`, `drive.mount(...)`), skip or remove those cells.
5. Run notebook cells in order.

## Current Modeling Coverage

- Popularity baseline
- SVD baseline
- Hybrid recommender variants
- LightGCN-based experimentation
- LLM-embedding-augmented neural recommender

## Notes

- This repository is research/experimentation oriented and notebook-driven.
- Results and model artifacts depend on data snapshot, preprocessing choices, and runtime environment.
- For reproducibility, consider adding a single `requirements.txt` and centralized config for dataset paths.
