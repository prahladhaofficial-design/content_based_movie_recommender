# Content-Based Movie Recommender

A command-line recommendation engine that uses movie metadata, TF-IDF features, and cosine similarity to suggest related titles.

## Overview

Content-based recommendation is a strong baseline when user interaction history is unavailable. This project recommends movies using the text and metadata already present in CSV files.
The implementation is simple enough to inspect and practical enough to run from the terminal.

## Project Design

- Load multiple movie datasets with different schemas.
- Detect the best available title column automatically.
- Build text features from titles, genres, ratings, and available metadata.
- Vectorize movies with TF-IDF and rank similar titles with cosine similarity.
- Expose listing and recommendation through a CLI.

## Key Capabilities

- Bollywood, Telugu, and IMDb-style CSV loading.
- Automatic title-column detection.
- Metadata-to-text feature construction.
- TF-IDF vectorization.
- Cosine-similarity recommendation ranking.
- Command-line interface for listing titles and requesting top-k recommendations.

## Repository Structure

```text
src/recommender.py             Recommendation engine and CLI.
data/bollywood_movies.csv      Bollywood movie metadata.
data/TeluguMovies_dataset.csv  Telugu movie metadata.
data/imdb_top_1000.csv         IMDb-style metadata.
requirements.txt               Python dependencies.
```

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/recommender.py --list
python src/recommender.py --title "Movie Title" --top-k 10
```

## Results and Artifacts

- Terminal output listing available titles.
- Top-k recommendations for a selected movie.
- Reusable recommendation logic in `src/recommender.py`.

## Reproducibility Notes

- Keep the movie metadata CSV in place before running the recommendation notebook.
- Rebuild TF-IDF features when changing text columns, cleaning rules, or similarity metrics.
- Use deterministic query examples when comparing recommendation quality across changes.
- Treat cosine similarity results as content-neighborhood suggestions, not collaborative-filtering predictions.

## Detailed Workflow

- Install dependencies and run the title-listing command first.
- Choose a title from the printed list to avoid spelling mismatch issues.
- Run the recommendation command with a selected title and top-k value.
- Inspect `src/recommender.py` to understand feature construction and similarity ranking.
- Try different datasets or metadata columns to change recommendation behavior.

## Implementation Details

- The recommender loads multiple movie CSV schemas and detects title columns automatically.
- Text features are built from available movie metadata.
- TF-IDF converts metadata text into sparse numeric vectors.
- Cosine similarity ranks movies by vector closeness.
- The CLI keeps the project usable without a notebook or web app.

## Data and Inputs

- The repository includes Bollywood, Telugu, and IMDb-style CSV files.
- Different datasets contain different metadata fields, so feature text depends on available columns.
- Ratings and genres can influence recommendations when they are present in the feature text.
- Adding richer metadata such as directors, actors, descriptions, or languages would improve recommendations.

## How To Interpret the Results

- Recommendations are based on content similarity, not user behavior.
- A movie can be recommended because it shares genre, title terms, rating context, or other metadata.
- TF-IDF is explainable but limited when metadata is sparse.
- Results should be reviewed as nearest-neighbor suggestions, not personalized rankings.

## Limitations and Assumptions

- The system does not use collaborative filtering.
- Misspelled titles may fail unless fuzzy matching is added.
- Sparse metadata can produce weaker recommendations.
- The CLI is intentionally simple and does not include a graphical interface.


## Review Guide

Use this checklist when reviewing the repository on GitHub:

- Start with the overview to understand the project objective before opening notebooks or source files.
- Read the repository structure section to identify the main entry point.
- Inspect generated outputs before rerunning experiments; they provide a quick view of the current project state.
- Review scripts and notebooks together when both are present, since notebooks often explain decisions while scripts preserve reusable execution paths.
- Check the limitations section before interpreting results too strongly.
- Use the possible extensions section to understand how the project can grow without changing its core purpose.

## Troubleshooting

Common issues and fixes:

- If imports fail, confirm that the virtual environment is activated and dependencies were installed from `requirements.txt`.
- If a notebook cell fails because a variable is missing, restart the kernel and run the notebook from the first cell.
- If a script cannot find a data file, run it from the repository root unless the README gives a different working directory.
- If plots are not regenerated, check that the output folder exists and that the script has write permission.
- If a package version causes errors, create a fresh environment and reinstall dependencies before debugging the source code.
- If GPU-related packages are unavailable, run CPU-compatible sections first and treat heavier deep-learning experiments separately.

## Development Notes

The repository is organized to be understandable before it is optimized. Most code favors clear structure, visible intermediate steps, and saved artifacts over highly compressed implementations. When extending the project, prefer small changes that keep the original experiment reproducible.

Recommended development practices:

- Keep new data files under `data/`, `datasets/`, or the existing local data folder for that project.
- Keep generated figures under `figures/`, `plots/`, `outputs/`, or `results/` depending on the existing convention.
- Add new experiments as separate scripts or clearly labeled notebook sections.
- Avoid overwriting existing outputs unless the experiment is intentionally being refreshed.
- Document any new dependency in `requirements.txt`.
- Prefer relative paths so the repository works after cloning on another machine.

## Project Status

This repository is prepared as a standalone GitHub project. The current version includes the core implementation, documentation, and available artifacts needed to understand the work. Future improvements can focus on stronger automation, more tests, richer visual summaries, and cleaner packaging where appropriate.

## Possible Extensions

- Add fuzzy title matching for misspelled queries.
- Add hybrid ranking using rating and year filters.
- Expose the recommender through a small web API.
- Add evaluation with held-out similarity labels if available.

## Tech Stack

Python, Pandas, scikit-learn, TF-IDF, cosine similarity.

## Author

Prahladha, IIT Hyderabad
