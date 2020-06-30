# Disaster Response Pipeline Project

This project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## File Description
- `app`
  - `run.py`  -  Flask application
  - `template/`
    - `master.html`  -  Main page of application
    - `go.html`  -  Model classification result of application
- `data`
  - `disaster_categories.csv`  -  Disaster categories dataset
  - `disaster_messages.csv`  -  Disaster messages dataset
  - `DisasterResponse.db`  -  The sqlite database with merged and cleaned data
  - `process_data.py`  -  The data processing pipeline Python script
- `models`
  - `train_classifier.py`  -  The NLP pipeline, trains with data and saves the model.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Github repo:
[https://github.com/carterjin/Disaster-Response-Pipeline](https://github.com/carterjin/Disaster-Response-Pipeline)
