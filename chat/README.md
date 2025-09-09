### Enter the python virtual environment

1. `cd Miscellaneous\KgGen\GraphJudge`
2. `.venv\Scripts\activate` 

### Graph Judge execute

- Standard mode (original behavior)
python run_gj.py

- Explainable mode (dual file output)
python run_gj.py --explainable

- Custom reasoning file path
python run_gj.py --explainable --reasoning-file custom_reasoning.json

### unit test

test command with coverage.json report:
`pytest <test file name>.py --cov=. --cov-report=json:<test file name>.json --cov-report=term -v`
