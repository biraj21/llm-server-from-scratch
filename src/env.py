from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    HF_TOKEN: str


# load from environment variables
env = Settings()  # type: ignore
