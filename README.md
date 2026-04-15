# 🦖 Dinosaur AI: Enterprise Edition

**Dinosaur AI** is a professional-grade ML platform for automated EDA and predictive modeling. Refactored for high-performance data engineering and production-ready AI pipelines.

## 🚀 Key Improvements (Senior Refactor)

- **MLOps Pipeline**: Implemented `sklearn.Pipeline` with `ColumnTransformer` to prevent data leakage.
- **Automated Cleaning**: New `DinoCleaner` engine with regex-based numeric extraction and column standardization.
- **Professional Architecture**: Decoupled configuration using `.env` and `config.py`.
- **Unit Testing**: 100% core coverage using `pytest`.
- **Premium Dashboard**: Glassmorphic HTML5 frontend for model visualization and real-time prediction simulation.

## 🛠 Tech Stack

- **Core**: Python 3.13, Pandas, Scikit-learn, Joblib.
- **Quality**: Pytest, Dotenv.
- **Frontend**: HTML5, CSS3 (Vanilla), Chart.js.

## 📦 Project Structure

```text
├── dinosaur/          # Core Library (Cleaner, Reader, Analyzer, Visualizer)
├── models/            # Serialized ML artifacts (.joblib)
├── tests/             # Unit and integration tests
├── .env               # External configuration
├── config.py          # Centralized settings
├── model_pipeline.py  # Production training pipeline
└── dashboard.html     # Premium UI
```

## 🚦 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn seaborn joblib python-dotenv pytest
   ```

2. **Run Pipeline**:
   ```bash
   python model_pipeline.py
   ```

3. **Run Tests**:
   ```bash
   pytest
   ```

4. **Launch Dashboard**:
   Open `dashboard.html` in any modern browser.

---
Created with 🦖 by the Senior AI Engineering Team.
