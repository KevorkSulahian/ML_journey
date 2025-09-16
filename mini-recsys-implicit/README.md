# Mini RecSys with Implicit Feedback

A minimal recommender system implementation using implicit feedback and the [implicit](https://github.com/benfred/implicit) library.

## Project Structure

```
mini-recsys-implicit/
├─ data/                  # raw dataset (ignored in git)
├─ artifacts/             # saved model + maps
├─ recsys/               # core package
│  ├─ data.py            # load/split, id maps
│  ├─ model_als.py       # train/save/load ALS
│  ├─ eval.py            # metrics
│  ├─ service.py         # FastAPI app
│  ├─ utils.py           # utilities
├─ scripts/              # executable scripts
│  ├─ download_movielens.py
│  ├─ train_als.py
│  ├─ evaluate.py
│  ├─ serve.py
├─ tests/                # unit tests
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Unix
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -e ".[test]"
```

3. Download the MovieLens dataset:
```bash
python scripts/download_movielens.py
```

## Usage

1. Train the model:
```bash
python scripts/train_als.py
```

2. Evaluate the model:
```bash
python scripts/evaluate.py
```

3. Run the service:
```bash
python scripts/serve.py
```

The service will be available at `http://localhost:8000`.

## API Endpoints

- `GET /health`: Health check
- `POST /recommend`: Get recommendations for a user
  ```json
  {
    "user_id": 123,
    "n_items": 10
  }
  ```

## Testing

Run tests with:
```bash
pytest
```