# config.py
from dotenv import load_dotenv
import os
from pathlib import Path

# Get the base directory of your project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
env_path = BASE_DIR / '.env'
load_dotenv(env_path)

# Database settings
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "claude_db")
DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Redis settings
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')

# Anthropic settings
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL')

# Print configuration (for debugging)
def print_config():
    print("Current configuration:")
    print(f"DATABASE_URL: {DATABASE_URL}")
    print(f"REDIS_HOST: {REDIS_HOST}")
    print(f"REDIS_PORT: {REDIS_PORT}")
    print(f"ANTHROPIC_API_KEY: {'Set' if ANTHROPIC_API_KEY else 'Not Set'}")
    print(f"ANTHROPIC_MODEL: {ANTHROPIC_MODEL}")