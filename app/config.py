import os
from dotenv import load_dotenv

load_dotenv()  # optional: loads .env in development

QUIZ_SECRET = os.getenv("QUIZ_SECRET", "")   # same as in Google Form
QUIZ_EMAIL = os.getenv("QUIZ_EMAIL", "")
MAX_TOTAL_SECONDS = int(os.getenv("MAX_TOTAL_SECONDS", "170"))  # < 180s buffer
