# Rain Prediction in Sa Pa, Vietnam

Midterm project: a machine learning web service that predicts whether it is **raining in Sa Pa (Vietnam)** from current weather sensor readings.

## 1. Problem description

Sa Pa is a mountain town in northern Vietnam and a popular trekking destination. Its weather changes fast: a clear morning can turn into heavy rain within an hour. Rain makes mountain trails slippery and causes landslides, so hikers, tour operators and local farmers need to know whether it is raining or likely to rain.

Weather stations and weather APIs report many numeric measurements (temperature, humidity, pressure, wind, clouds, visibility). A rain label is not always available, or comes late. This project asks:

> **Given the current weather measurements in Sa Pa, is it raining?** (binary classification: `rain` / `no rain`)

**Dataset.** [Sapa Weather (Kaggle)](https://www.kaggle.com/datasets/evgenyarbatov/sapa-weather) contains weather observations recorded every 30 minutes from 25 Aug 2026 to 25 Sep 2026 (1511 rows, 20 columns).

- **Target:** `weather_main` (`Clear`, `Clouds`, `Rain`), turned into binary: `Rain` → 1, otherwise → 0. Only about 13% of the rows are rain, so the classes are imbalanced. Models are compared with **ROC AUC**, not accuracy.
- **Features used (9):** `weather_temp`, `weather_feels_like`, `weather_humidity`, `weather_pressure`, `weather_wind_speed`, `weather_wind_deg`, `weather_wind_gust`, `weather_clouds`, `weather_visibility`.
- **Dropped columns:** location and metadata (`lat`, `lon`, `site_id`, `source`, `schema_version`), timestamps, and `weather_rain_1h`, `weather_id`, `weather_description`. The last three directly reveal the target (data leakage).
- **Missing values:** `weather_visibility` has a few missing values, which are filled with the column mean.

**Solution.** Several models were trained and compared: logistic regression, decision tree, random forest and XGBoost. The best one is a **Random Forest** (`n_estimators=23`, `max_depth=8`), chosen with a grid search over `n_estimators` × `max_depth`. It reaches a mean 8-fold cross-validation AUC of about **0.86** and a test AUC of about **0.89**.

**How it is used.** The model is served as a REST API. A client (a weather dashboard, a trekking app, a chatbot) sends the current weather readings as JSON and gets back the probability of rain and a yes/no decision. For example, the app can warn trekkers when `rain` is `true`.

## 2. Repository structure

```
Midterm_project/
├── README.md                     <- this file
├── Data/
│   └── README.md                 <- how to get the dataset
├── Notebook/
│   └── notebook.ipynb            <- data cleaning, EDA, feature importance, model selection & tuning
├── Script/
│   ├── train.py                  <- trains the final model and saves it to model.bin
│   ├── predict.py                <- Flask web service that loads model.bin and serves /predict
│   ├── send_request.py           <- example client that calls the service
│   ├── model.bin                 <- trained model (DictVectorizer + RandomForest, pickled)
│   ├── requirements.txt          <- pinned dependencies
│   └── Dockerfile                <- container for the prediction service
└── Docker deployment video/
    └── docker deploy.rar         <- video of the deployed service in use (unzip to watch)
```

## 3. Data

The dataset is downloaded **automatically** by both `notebook.ipynb` and `train.py` with [`kagglehub`](https://github.com/Kaggle/kagglehub):

```python
import kagglehub
path = kagglehub.dataset_download("evgenyarbatov/sapa-weather")
```

The dataset is public, so no Kaggle account or API key is needed. To download it by hand instead, go to https://www.kaggle.com/datasets/evgenyarbatov/sapa-weather and get `sapa-observations.parquet`.

> Note: the Kaggle dataset is still being updated. The results in the notebook were produced with the version that had 1511 rows (25 Aug – 25 Sep 2026). A newer version may give slightly different scores.

## 4. How to run the project

Requirements: **Python 3.13**, plus **Docker** for the container part. The commands below are run from the project root unless stated otherwise.

### 4.1 Create and activate a virtual environment, then install dependencies

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r Script/requirements.txt
```

The notebook also needs Jupyter and the plotting and XGBoost libraries:

```bash
pip install jupyter matplotlib seaborn xgboost
```

### 4.2 Run the notebook

```bash
jupyter notebook Notebook/notebook.ipynb
```

The notebook was developed on Google Colab. It can also be opened directly in Colab, where all the libraries are pre-installed.

### 4.3 Train the model

```bash
cd Script
python train.py
```

This downloads the data, runs 8-fold cross-validation, prints the mean validation AUC and the test AUC, and writes `model.bin`.

### 4.4 Run the web service locally

`predict.py` loads `model.bin` and exposes `POST /predict` on port **9000**.

On Windows, run the development server:

```bash
cd Script
python predict.py
```

On Linux / macOS, you can use gunicorn (it does not run on Windows):

```bash
cd Script
gunicorn --bind=0.0.0.0:9000 predict:app
```

### 4.5 Send a request

Request body: one JSON object with the 9 weather features.

```bash
curl -X POST http://localhost:9000/predict -H "Content-Type: application/json" -d '{"weather_temp": 16.74, "weather_feels_like": 16.72, "weather_humidity": 86.0, "weather_pressure": 1013.0, "weather_wind_speed": 0.38, "weather_wind_deg": 42.0, "weather_wind_gust": 0.8, "weather_clouds": 49.0, "weather_visibility": 10000.0}'
```

Response:

```json
{"rain": false, "rain prob": 0.21}
```

- `rain prob`: predicted probability of rain (0–1)
- `rain`: `true` when the probability is ≥ 0.5

(The exact probability depends on the trained model.)

You can also use the example client:

```bash
cd Script
python send_request.py
```

`send_request.py` sends its request to `http://host.docker.internal:9000/predict`. This works on Windows / macOS with Docker Desktop. On Linux, or when the service runs without Docker, change the URL to `http://localhost:9000/predict`.

## 5. Containerization (Docker)

The `Dockerfile` is in `Script/`. It uses `python:3.13.9-slim`, installs `requirements.txt`, copies the model and the service code, and starts the app with **gunicorn** on port 9000.

Build the image:

```bash
cd Script
docker build -t sapa-rain-prediction .
```

Run the container (maps container port 9000 to host port 9000):

```bash
docker run -it --rm -p 9000:9000 sapa-rain-prediction
```

Then send a request from another terminal (see section 4.5):

```bash
cd Script
python send_request.py
```

## 6. Deployment

The service was deployed in a Docker container and tested with requests. A screen recording is in [`Docker deployment video/docker deploy.rar`](Docker%20deployment%20video/). Unzip it (for example with 7-Zip or WinRAR) to watch the video.

## 7. Results summary

| Model | Validation AUC |
|---|---|
| Logistic regression (baseline) | ~0.77 |
| Decision tree (`max_depth=4`) | ~0.78 |
| XGBoost (`eta=0.3`, `max_depth=7`) | ~0.81 |
| Random forest (`n_estimators=100`, `max_depth=6`) | ~0.84 |
| **Random forest, tuned (`n_estimators=23`, `max_depth=8`)** | **best on the grid search** |

Final model: 8-fold CV mean AUC ≈ 0.86, test AUC ≈ 0.89.

