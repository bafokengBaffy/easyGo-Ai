import redis
from .settings import settings

redis_client = redis.from_url(settings.redis_url)
