# Health Insurance ML API

This is a Flask-based backend for training, testing, and predicting health insurance charges.

## Endpoints

1. **/train** (POST) - Upload a CSV file with training data to train the model.
2. **/test** (POST) - Upload a CSV file with test data to evaluate the model.
3. **/predict** (POST) - Send a JSON payload to get a prediction.

## Usage

### Train the model
```bash
curl -X POST -F "file=@your_dataset.csv" http://127.0.0.1:5000/train
```

### Test the model
```bash
curl -X POST -F "file=@your_dataset.csv" http://127.0.0.1:5000/test
```

### Predict
```bash
curl -X POST -H "Content-Type: application/json" -d '{"age": 45, "sex": 1, "bmi": 28.5, "children": 2, "smoker": 0, "region": 2}' http://127.0.0.1:5000/predict
```

## Install & Run
```bash
pip install -r requirements.txt
python app.py
```
