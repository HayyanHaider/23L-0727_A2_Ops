from dotenv import load_dotenv

load_dotenv()
from visionguard.web import create_app  # noqa: E402

app = create_app()
