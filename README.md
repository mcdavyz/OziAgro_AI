# OziAgro Farmer-Centred Climate Decision-Support System

## Prototype v1.0

OziAgro is a farmer-centred climate decision-support prototype for
rainfed rice production. The system estimates short-term dry-spell
risk from rainfall conditions and translates the predicted risk into
an irrigation-aware planting recommendation.

### Prototype Study Area

Bende Local Government Area, Abia State, Nigeria.

### Core Decision Problem

Given rainfall conditions at a prediction origin, is there an
elevated probability of a dry spell of at least seven consecutive
dry days occurring within the next 14 days, and what planting action
should be considered based on irrigation availability?

## System Workflow

Historical rainfall
→ Feature engineering
→ XGBoost prediction
→ Dry-spell probability
→ Risk classification
→ Irrigation-aware recommendation

## Prototype Definitions

- Dry day: rainfall < 1.0 mm
- Dry spell: ≥7 consecutive dry days
- Prediction horizon: 14 days
- Modelling period: DOY 60–309
- Historical rainfall period: 1981–2025
- Operational model: XGBoost
- DSS probability threshold: 0.45

## Predictor Variables

The prototype uses 15 rainfall-derived predictors:

- Consecutive dry days
- Rainfall totals over 1, 3, 5, 7, 14 and 30 days
- Rainy-day counts over 7, 14 and 30 days
- Dry-day counts over 7, 14 and 30 days
- Cyclical day-of-year variables (`doy_sin`, `doy_cos`)

## Data

The prototype uses historical daily rainfall derived from CHIRPS
v2.0 for Bende LGA covering 1981–2025.

The historical dataset is used for model development and prototype
testing. Live climate-data retrieval is not yet part of this
prototype.

## Model Development

Three classification approaches were developed:

1. Logistic Regression
2. Random Forest
3. XGBoost

The data were divided chronologically:

| Dataset | Period |
|---|---|
| Training | 1981–2014 |
| Validation | 2015–2019 |
| Independent test | 2020–2025 |

The operational XGBoost threshold was selected using the validation
dataset and subsequently evaluated on the independent test dataset.

## DSS Logic

| Risk | Irrigation | Recommendation |
|---|---|---|
| Low | No | Planting can proceed. |
| Low | Yes | Planting can proceed; irrigation can serve as backup. |
| High | Yes | Planting may proceed with supplemental irrigation. |
| High | No | Delay planting until dry-spell risk decreases. |

## Prototype Test Performance

At the locked 0.45 threshold on the independent test dataset:

- Precision: 0.1172
- Recall: 0.2679
- Specificity: 0.9217
- F1-score: 0.1630
- False-plant rate: 0.0783
- False-wait rate: 0.7321

## Project Structure

```text
OziAgro_AI/
│
├── data/
├── docs/
│   └── OziAgro_Prototype_Documentation.md
├── outputs/
├── scripts/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── target_creation.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── dss_logic.py
│   └── prediction_service.py
│
├── app.py
├── main.py
├── predict.py
├── test_prediction.py
├── requirements.txt
└── README.md

Running the Prototype

Activate the Python virtual environment and install the required
packages: 
pip install -r requirements.txt

Run the prediction pipeline:
python predict.py

Launch the Streamlit interface:
streamlit run app.py
