from typing import Any, Dict, List, Optional, Type, cast

from collections import defaultdict

from bentoml import BentoService, config
from prometheus_client import Counter, Histogram
from prometheus_client.context_managers import ExceptionCounter
from prometheus_client.metrics import MetricWrapperBase

from bento2seldon.seldon import DEPLOYMENT_ID
from bento2seldon.utils import Timer

PREDICT_ENDPOINT = "predict"
FEEDBACK_ENDPOINT = "send-feedback"


class Monitor:
    def __init__(self, bento_service: BentoService) -> None:
        self.service_name = bento_service.name
        self.version = bento_service.version
        self.namespace = config("instrument").get("default_namespace")

    def _create_metric(
        self,
        metric_class: Type[MetricWrapperBase],
        suffix: str,
        documentation: Optional[str] = None,
        labelnames: Optional[List[str]] = None,
    ) -> MetricWrapperBase:
        labelnames = [
            "deployment_id",
            "service_version",
            "endpoint",
            *(labelnames or []),
        ]
        return metric_class(
            name=f"{self.service_name}_{suffix}",
            namespace=self.namespace,
            documentation=documentation or "",
            labelnames=labelnames,
        )

    def count_exceptions(
        self, endpoint: str = PREDICT_ENDPOINT, extra: Optional[Dict[str, Any]] = None
    ) -> ExceptionCounter:
        if extra is None:
            extra = {}

        if not hasattr(self, "_exception_counter"):
            self._exception_counter = self._create_metric(
                Counter,
                "exception_total",
                "Total number of exceptions",
                list(extra.keys()),
            )

        return cast(
            Counter,
            self._exception_counter.labels(
                DEPLOYMENT_ID, self.version, endpoint, *extra.values()
            ),
        ).count_exceptions()

    def time_model_execution(
        self,
        parallel_executions: int,
        endpoint: str = PREDICT_ENDPOINT,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Timer:
        if extra is None:
            extra = {}

        if not hasattr(self, "_model_execution_duration"):
            self._model_execution_duration = self._create_metric(
                Histogram,
                "model_execution_duration_seconds",
                "Model execution duration in seconds",
                list(extra.keys()),
            )
        if not hasattr(self, "_model_execution_per_request_duration"):
            self._model_execution_per_request_duration = self._create_metric(
                Histogram,
                "model_execution_per_request_duration_seconds",
                "Model execution per request duration in seconds",
                list(extra.keys()),
            )

        def observe(duration: float) -> None:
            cast(
                Histogram,
                self._model_execution_duration.labels(
                    DEPLOYMENT_ID, self.version, endpoint, *list(extra.values())  # type: ignore[union-attr]
                ),
            ).observe(duration)
            cast(
                Histogram,
                self._model_execution_per_request_duration.labels(
                    DEPLOYMENT_ID, self.version, endpoint, *list(extra.values())  # type: ignore[union-attr]
                ),
            ).observe(duration / parallel_executions)

        return Timer(observe)

    def observe_reward(
        self,
        value: float,
        endpoint: str = FEEDBACK_ENDPOINT,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if extra is None:
            extra = {}

        if not hasattr(self, "_reward"):
            self._reward = self._create_metric(
                Histogram, "reward", "Reward provided by feedback", list(extra.keys())
            )

        cast(
            Histogram,
            self._reward.labels(DEPLOYMENT_ID, self.version, endpoint, *extra.values()),
        ).observe(value)
