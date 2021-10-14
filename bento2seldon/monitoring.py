from typing import List, Optional, Type

from bentoml import BentoService, config
from prometheus_client import Counter, Histogram
from prometheus_client.context_managers import ExceptionCounter, Timer
from prometheus_client.metrics import MetricWrapperBase

from bento2seldon.seldon import DEPLOYMENT_ID

PREDICT_ENDPOINT = "predict"
FEEDBACK_ENDPOINT = "send-feedback"


class Monitor:
    def __init__(self, bento_service: BentoService) -> None:
        self.service_name = bento_service.name
        self.version = bento_service.version
        self.namespace = config("instrument").get("default_namespace")

        self._exception_counter = self._create_metric(
            Counter, "exception_total", "Total number of exceptions"
        )

        self._model_execution_duration = self._create_metric(
            Histogram,
            "model_execution_duration_seconds",
            "Model execution duration in seconds",
        )

        self._model_execution_per_request_duration = self._create_metric(
            Histogram,
            "model_execution_per_request_duration_seconds",
            "Model execution per request duration in seconds",
        )

        self._reward = self._create_metric(
            Histogram, "reward", "Reward provided by feedback"
        )

    def _create_metric(
        self,
        metric_class: Type[MetricWrapperBase],
        suffix: str,
        documentation: Optional[str] = None,
        labelnames: Optional[List[str]] = None,
    ) -> MetricWrapperBase:
        if labelnames is None:
            labelnames = []
        labelnames = [
            "deployment_id",
            "service_version",
            "endpoint",
            *labelnames,
        ]
        return metric_class(
            name=f"{self.service_name}_{suffix}",
            namespace=self.namespace,
            documentation=documentation,
            labelnames=labelnames,
        )

    def count_exceptions(self, endpoint: str = PREDICT_ENDPOINT) -> ExceptionCounter:
        return self._exception_counter.labels(
            DEPLOYMENT_ID,
            self.version,
            endpoint,
        ).count_exceptions()

    def time_model_execution(
        self, parallel_executions: int, endpoint: str = PREDICT_ENDPOINT
    ) -> Timer:
        def observe(duration: float) -> None:
            self._model_execution_duration.labels(
                DEPLOYMENT_ID, self.version, endpoint
            ).observe(duration)
            self._model_execution_per_request_duration.labels(
                DEPLOYMENT_ID, self.version, endpoint
            ).observe(duration / parallel_executions)

        return Timer(observe)

    def observe_reward(self, value: float, endpoint: str = FEEDBACK_ENDPOINT) -> None:
        self._reward.labels(DEPLOYMENT_ID, self.version, endpoint).observe(value)
