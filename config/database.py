from asyncpg import connect, Connection
from .settings import settings

db: Connection | None = None

async def create_db_connection() -> Connection:
    global db
    if db is None:
        db = await connect(settings.database_url)
    return db

async def close_db_connection() -> None:
    global db
    if db is not None:
        await db.close()
        db = None
