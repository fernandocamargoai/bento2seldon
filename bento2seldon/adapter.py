import logging

from bentoml.adapters import JsonInput
from bentoml.types import HTTPRequest, InferenceTask

logger = logging.getLogger(__name__)


class SeldonJsonInput(JsonInput):
    def from_http_request(self, req: HTTPRequest) -> InferenceTask[str]:
        inference_task = super().from_http_request(req)
        puid = req.headers.get("Seldon-Puid")
        if puid:
            logger.debug("Setting the task_id from the puid header: %s", puid)
            inference_task.task_id = puid
        return inference_task
