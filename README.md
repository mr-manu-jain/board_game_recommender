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
- `users/manu/eda-manujain.ipynb`: Additional EDA and data checks.
- `users/manu/baseline-popularitybased-manu.ipynb`: Popularity-based baseline recommender.
- `users/manu/baseline-svd-manu.ipynb`: SVD-based collaborative filtering baseline.
- `users/manu/llm-embedding.ipynb`: LLM-embedding + hybrid model workflow.
- `users/rahul/EDA_Rahul.ipynb`: EDA variant by Rahul.
- `users/rahul/Hybrid_recommender_Rahul.ipynb`: Hybrid recommender experiment.
- `users/darshan/darshan_lightgcn.ipynb`: LightGCN experiment.
- `users/manu/nnw`: Neural recommender workflow (JSON notebook export format).

## Data Overview

The notebooks reference a BGG-derived dataset with:

- Game metadata (`games.csv`) and structured feature tables (`themes`, `mechanics`, `designers`, `artists`, `publishers`, etc.).
- User interactions (`user_ratings.csv`) with tens of millions of rating records.
- Processed parquet/csv artifacts for training and evaluation splits in some workflows.

Most notebooks currently expect files mounted from a shared Google Drive path:

`/content/drive/Shareddrives/CMPE256_FinalProject/board_game_recommendation`

If you run locally, update the path variables in each notebook to point to your data location.

## Environment Setup

Recommended: Python 3.10+ and Jupyter/Colab.

1. Create and activate a virtual environment.
2. Install core dependencies:

```bash
pip install jupyter polars pandas numpy scipy scikit-learn matplotlib seaborn torch pyarrow
```

Optional (depending on notebook):

- `surprise` for matrix-factorization style baselines.
- GPU-enabled PyTorch build for faster training.

## Running the Project

### Option 1: Google Colab (matches current notebook paths)

1. Upload/open notebooks in Colab.
2. Mount Google Drive.
3. Confirm the dataset path in the first cells.
4. Run cells top-to-bottom.

### Option 2: Local Jupyter

1. Place raw/processed data in a local directory.
2. Open each notebook and replace Google Drive paths with local paths.
3. Run notebook cells in order.

## Current Modeling Coverage

- Popularity baseline
- SVD baseline
- Hybrid recommender variants
- LightGCN-based experimentation
- LLM-embedding-augmented neural recommender

## Notes

- This repository is research/experimentation oriented and notebook-driven.
- Results and model artifacts depend on data snapshot, preprocessing choices, and runtime environment.
- To improve reproducibility, consider adding a single `requirements.txt` and centralized config for dataset paths.
