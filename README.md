# 🦖 Dinosaur EDA

**Dinosaur** is a Python library designed to automate and simplify Exploratory Data Analysis (EDA). It's built on top of `pandas`, `matplotlib`, and `seaborn`.

## Features

- **Smart Loading**: Automatically detects file formats (CSV, Excel, JSON, etc.).
- **Auto-Reports**: Get a data quality report with one command.
- **Visual Intelligence**: Automated generation of distribution plots, correlation heatmaps, and missing value matrices.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import dinosaur as dino

# Load and explore in one line
explorer = dino.explore("your_dataset.csv")

# Or more control
data = dino.Dinosaur("data.xlsx")
data.report()
data.visualize(mode='correlation')
```

## Authors
Created with 🦖 for data scientists.
