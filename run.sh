python -m venv ./.vantage6_venv

source ./.vantage6_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
