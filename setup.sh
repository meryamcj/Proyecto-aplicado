# setup.sh

# Install Python
sudo apt-get update
sudo apt-get install python3

python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt