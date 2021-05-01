from typing import Optional

from pydantic import BaseSettings, RedisDsn


class Settings(BaseSettings):
    redis_url: Optional[RedisDsn]
    cache_duration: int = 24 * 60 * 60
