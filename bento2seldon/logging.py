from typing import Any, Dict

import uuid
from copy import deepcopy

from bentoml import BentoService


class LoggingContext(Dict[str, Any]):
    def __init__(self, bento_service: BentoService) -> None:
        super().__init__()

        self["service"] = {
            "name": bento_service.name,
            "version": bento_service.version,
        }
        self["http.request.uuid"] = str(uuid.uuid4())

    def with_endpoint(self, endpoint: str) -> "LoggingContext":
        self["http.request.endpoint"] = endpoint
        return self

    def with_batch_size(self, batch_size: int) -> "LoggingContext":
        self["http.request.batch_size"] = batch_size
        return self

    def with_cache_hits(self, cache_hits: int) -> "LoggingContext":
        self["http.response.cache_hits"] = cache_hits
        return self

    def with_status(self, status: int) -> "LoggingContext":
        context = deepcopy(self)
        context["http.response.status"] = status
        return context
