# Crop Recommendation Based on Soil and Climate Conditions

**YEEAP Capstone · FUSE AI-201 – AI Model Development and Implementation · Modules 5–6 · Group 6 · Domain: Agriculture**

Given a plot's soil nutrients (N, P, K), soil pH and local climate (temperature, humidity, rainfall), the system recommends the most suitable crop out of 22 candidates and shows the next two best alternatives with confidence scores. It is a **decision-support demo** to be confirmed with a local extension officer, not a validated advisory service.

**GitHub repository:** https://github.com/beby2846/crop-recommendation-group6
**Live app (Streamlit):** not deployed yet

```
Problem → Data → Analysis → Model → Evaluation → Testing → Working Prototype
```

## Results at a glance (held-out test set, 330 rows)

| Metric | Target (Module 4) | Result |
|---|---|---|
| Macro F1 | ≥ 0.90 | **0.997** |
| Accuracy | ≥ 0.90 | **0.997** (1 error in 330) |
| Lowest per-crop recall | ≥ 0.80 | **0.933** (rice) |
| Top-3 accuracy | ≥ 98 % | **100 %** |
| Beats baseline (KNN, macro F1 0.972) | yes | **+0.025** (95 % CI +0.009 to +0.045) |
| Single recommendation latency | < 1 s | **≈ 10 ms** |

Final model: **Random Forest** (as proposed). All numbers come from running `crop_recommendation_notebook.ipynb`; see the report for interpretation and limitations (clean benchmark, India-specific data, sensitivity to noisy readings).

## Project structure

```
crop_project/
├── crop_recommendation_notebook.ipynb   # Tasks 1–8 in order (already executed; re-runnable)
├── Project_Report.docx                  # Written report: process, results, conclusions
├── README.md                            # this file
├── requirements.txt                     # Python dependencies
├── app.py                               # Streamlit prototype (the working application)
├── recommender.py                       # shared input validation + prediction code
├── data/Crop_recommendation.csv         # dataset (2,200 rows, 7 features, 22 crops)
├── models/crop_model.joblib             # saved final model (scikit-learn pipeline)
├── models/model_meta.json               # classes, training ranges, importances, metrics
├── outputs/metrics.json                 # all reported numbers, written by the notebook
├── outputs/test_cases.csv               # Task 8 test cases and outcomes
└── figures/                             # charts produced by the notebook (used in the report)
```

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

**1. Notebook (all eight tasks)** – open and *Run All*:

```bash
jupyter notebook crop_recommendation_notebook.ipynb
```

It reads `data/Crop_recommendation.csv`, and re-creates the figures, the saved model (`models/`) and `outputs/`. Seeds are fixed (42), so results are reproducible. Runtime is a few minutes on a laptop; no GPU needed.

**2. Web app (working prototype)**:

```bash
streamlit run app.py
```

Open the link shown in the terminal (usually http://localhost:8501), optionally pick an example plot, enter the seven readings and press **Recommend a crop**. Invalid values (e.g. humidity above 100 %) are rejected with a message; valid values outside the range seen in training are accepted but flagged with a warning.

**3. Use the model from Python**:

```python
from recommender import recommend
recommend({"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82, "ph": 6.5, "rainfall": 200})
# -> {'ok': True, 'top': [{'crop': 'rice', 'confidence': ...}, ...], 'warnings': [], 'errors': []}
```

**Troubleshooting:** if loading `models/crop_model.joblib` fails with a scikit-learn version error, run the notebook once to regenerate the model with your installed version.

## Module 6 – deployment, monitoring and maintenance additions

| File / folder | Purpose |
|---|---|
| `Module6_Documentation.docx` | Portfolio documentation: demo, testing evidence, experiment tracking, deployment approach, **monitoring / support / maintenance plan**, limitations, **simulated-data statement** |
| `One_Page_Retrospective.docx` | Deliverable C (add one sentence on real teamwork challenges where marked) |
| `Video_Script_and_Run_Sheet.docx` | Run sheet for the ≤ 10 min video (Deliverable B) with a presenter per member |
| `api.py` | FastAPI service: `POST /predict`, `GET /health`, `GET /model-info` (docs at `/docs`) |
| `track_experiments.py`, `mlflow.db`, `mlruns/`, `outputs/experiment_log.csv` | MLflow experiment tracking (6 runs) |
| `tests/` and `outputs/pytest_results.txt` | 29 automated tests (all passed) |
| `monitoring/drift_check.py`, `monitoring/simulate_and_check.py` | Drift / prediction-health check and its demo on **simulated** data |
| `Dockerfile`, `docker-compose.yml`, `requirements-deploy.txt` | Container deployment files (**written but not built/tested** where they were prepared) |
| `.github/workflows/ci.yml` | Runs the tests on every push once the repository is on GitHub |
| `screenshots/` | App, warning case and MLflow screenshots |

```bash
python -m pytest tests -q                                   # run the 29 tests
uvicorn api:app --port 8000                                  # REST API  (http://localhost:8000/docs)
python track_experiments.py                                  # re-log experiments in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db            # browse runs at http://localhost:5000
python monitoring/simulate_and_check.py                      # monitoring demo (simulated data)
python monitoring/drift_check.py new_readings.csv            # check real incoming readings
docker build -t crop-recommender . && docker run -p 8000:8000 crop-recommender   # untested
```

**Simulated data statement.** The model uses **real** data (Kaggle Crop Recommendation). Simulated data are used **only** in the monitoring demo (`outputs/simulated_*.csv`), built from the held-out test rows to show that drift checks work. Results on them are not evidence of real-world performance. No passwords, keys or confidential data are included.

**Not yet done:** public hosting (Streamlit Community Cloud link to be added here: _add link_), building the Docker image, a scheduled monitoring service.

## Data

- **Source:** Kaggle *Crop Recommendation Dataset* (<https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>). The CSV in `data/` is an identical copy taken from a public GitHub mirror (the fallback in proposal Section 3); the notebook verifies it has the expected 2,200 rows, 8 columns and 22 crops × 100 rows. You can replace it with the file downloaded from Kaggle.
- **Licence:** the proposal states the licence terms must be confirmed on the Kaggle listing before use – please do this before publishing the project.
- **Limitations:** built for Indian agronomic conditions; contains no personal data; does not include prices, irrigation or farmer preference.

## Changes from the Module 4 proposal

No change to domain, task, dataset, features or success criteria. The final-model choice was left to the data (Random Forest, SVM and Naive Bayes were compared); Random Forest was selected, as proposed. Details are in the notebook (Tasks 1 and 6) and the report.

## Group members and contributions

Responsibilities are taken from the group's Module 4 proposal (Sections 11 and 12). Before submitting, remove anyone who did not actually participate in Module 5 and adjust the wording if a member's real contribution differs.

**By area**

| Area | Responsible member(s) | Expected contribution |
|---|---|---|
| Domain / problem research | Henok Afework Birhanu; Shumye Birara Getu | Research precision-agriculture practices and support Section 1–2 evidence |
| Data discovery / preparation | Zinashwork Belay; Misgana Belete Songaye | Acquire dataset, run EDA, check data quality and splits |
| ML/DL development | Nigist Tesfu Berhane; Eden Beyene | Build baseline and candidate models, run evaluation and tuning |
| Application implementation | Binyam; Daniel Alemayehu Belachew | Build the form/interface that presents recommendations |
| Deployment / testing | Tsegaye Teka Bireda | Set up hosting/demo environment and test the end-to-end flow |
| Documentation / presentation | Tadele Tsedalu Birhane | Compile proposal/report content and prepare final demonstration materials |

**By stage**

| Stage | Key activities | Expected output | Responsible |
|---|---|---|---|
| Module 4: Proposal | Define problem, data, ML/DL task, MVP, roadmap | Approved project proposal | Henok Afework Birhanu; Shumye Birara Getu |
| Module 5: Data & baseline | Preparation, EDA, baseline model | Prepared data + baseline | Zinashwork Belay; Misgana Belete Songaye |
| Module 5: Model development | Training, evaluation, improvement | Selected model | Nigist Tesfu Berhane; Eden Beyene |
| Module 5: Implementation | Application integration and testing | Working AI application | Binyam; Shumye Birara Getu |
| Module 6: Deployment | Serving, deployment, configuration | Deployed/pilot system | Tsegaye Teka Bireda; Daniel Alemayehu Belachew |
| Module 6: Validation | Testing, validation, performance checks | Validated solution | Tsegaye Teka Bireda; Tadele Tsedalu Birhane |
| Final demonstration | Presentation and documentation | Capstone deliverables | Whole group |
