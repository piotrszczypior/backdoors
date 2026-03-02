python -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt

if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please update .env with your configuration (e.g., WANDB_API_KEY)."
fi
