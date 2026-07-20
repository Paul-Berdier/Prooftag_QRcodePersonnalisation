import logging
import time

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from .config import get_settings

logger = logging.getLogger(__name__)


def wait_for_database(database_url: str, attempts: int = 60, interval: float = 2.0) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    for attempt in range(1, attempts + 1):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(interval)


def main() -> None:
    settings = get_settings()
    wait_for_database(settings.database_url)
    config = Config("alembic.ini")
    command.upgrade(config, "head")


if __name__ == "__main__":
    main()
