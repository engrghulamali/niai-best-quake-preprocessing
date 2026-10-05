# Day 1 Quake Preprocessing

A Python-based earthquake data preprocessing project that fetches, cleans, transforms, and streams USGS seismic data into a model-ready dataset for analysis and machine learning.

## Overview

This project takes live earthquake feed data from the USGS API, converts it into structured tabular data, removes invalid or duplicate records, engineers useful features, and prepares the dataset for downstream modeling tasks. It is designed for learning and experimentation in data preprocessing, feature engineering, and pipeline building.

## Features

- Fetch earthquake data from the live USGS feed
- Fall back to local cached data when the API is unavailable
- Parse GeoJSON into a pandas DataFrame
- Remove duplicate and invalid records
- Convert timestamps and normalize event metadata
- Engineer time, quality, and location features
- Filter leakage-prone columns before modeling
- Save processed train/test data and model preprocessor artifacts

## Project Structure

```text
.
├── quakes/
│   ├── cleaning.py
│   ├── config.py
│   ├── features.py
│   ├── fetch.py
│   ├── run.py
│   ├── stream.py
│   └── transform.py
├── data/
│   ├── fallback/
│   ├── processed/
│   └── stream/
├── tests/
├── environment.yml
├── conftest.py
├── LAB_GUIDE.md
├── ANSWERS.md
├── README.md
└── .gitignore
```

## Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate niai-ml
```

## Run Tests

```bash
pytest
```

You can also run a specific test file:

```bash
pytest tests/test_fetch.py
pytest tests/test_cleaning.py
pytest tests/test_features.py
pytest tests/test_stream.py
pytest tests/test_transform.py
```

## Run the Pipeline

```bash
python -m quakes.run
```

This pipeline fetches the data, cleans it, merges stream updates, creates engineered features, splits the dataset, and saves processed outputs in `data/processed/`.

## Data Source

The project uses the USGS earthquake feed:

https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson

## GitHub

https://github.com/engrghulamali

## License

This project is intended for educational and learning purposes.
