from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import abc
import datetime
import logging

from bentoml import BentoService, api
from bentoml.types import InferenceTask
from decorator import decorate
from pydantic import ValidationError
from pydantic.main import BaseModel
from pydantic.tools import parse_obj_as

from bento2seldon.adapter import SeldonJsonInput
from bento2seldon.cache import Cache
from bento2seldon.logging import LoggingContext
from bento2seldon.model import Settings
from bento2seldon.monitoring import Monitor
from bento2seldon.seldon import (
    PRED_UNIT_ID,
    DefaultData,
    Feedback,
    Meta,
    RoutingSeldonMessage,
    SeldonMessage,
    SeldonMessageRequest,
    Status,
    StatusFlag,
    Tensor,
)

RT = TypeVar("RT", bound=BaseModel)
RE = TypeVar("RE", bound=BaseModel)
I = TypeVar(
    "I",
    bound=Union[
        SeldonMessage[BaseModel],
        SeldonMessageRequest[BaseModel],
        List[SeldonMessage[BaseModel]],
        List[SeldonMessageRequest[BaseModel]],
        Feedback[BaseModel, BaseModel],
    ],
)


logger = logging.getLogger(__name__)


class ExceptionHandler:
    def __init__(
        self,
        tasks: List[InferenceTask],
        logging_context: LoggingContext,
    ):
        self._tasks = tasks
        self._logging_context = logging_context

    def __enter__(self):
        pass

    def __exit__(self, typ, value, traceback):
        if isinstance(value, Exception):
            logger.exception(
                "Unexpected error", extra=self._logging_context.with_status(500)
            )
            error = SeldonMessage[Any](
                status=Status(
                    code=500, info=str(value), status=StatusFlag.FAILURE.value
                )
            )
            for task in self._tasks:
                task.discard(http_status=500, data=error.json(exclude_none=True))

    def __call__(self, f):
        def wrapped(func, *args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate(f, wrapped)


class BaseBentoService(BentoService, metaclass=abc.ABCMeta):
    def versioneer(self):
        return datetime.datetime.utcnow().strftime("%Y%m%d%H%M%I")

    def get_logger_context(
        self, endpoint: str = "predict", batch_size: Optional[int] = None
    ) -> LoggingContext:
        context = LoggingContext(self).with_endpoint(endpoint)
        if batch_size:
            context = context.with_batch_size(batch_size)
        return context

    @property
    def settings(self) -> Settings:
        if not hasattr(self, "_settings"):
            self._settings = Settings()
        return self._settings

    @property
    def monitor(self) -> Monitor:
        if not hasattr(self, "_monitor"):
            self._monitor = Monitor(self)
        return self._monitor

    def _parse_input(
        self,
        raw_request: Union[Dict[str, Any], List[Dict[str, Any]]],
        task: InferenceTask,
        request_type: Type[I],
        logging_context: LoggingContext,
    ) -> Optional[I]:
        try:
            input = parse_obj_as(request_type, raw_request)
            if isinstance(input, SeldonMessage):
                logger.debug(
                    "Setting the puid from task_id: %s",
                    task.task_id,
                    extra=logging_context,
                )
                input.meta.puid = task.task_id
            return input
        except ValidationError as e:
            logger.exception(
                "Validation error for input: %s",
                raw_request,
                extra=logging_context.with_status(400),
            )
            error = SeldonMessage[Any](
                status=Status(code=400, info=str(e), status=StatusFlag.FAILURE.value)
            )
            task.discard(http_status=400, data=error.json(exclude_none=True))
            return None

    def _parse_inputs(
        self,
        raw_requests: List[Dict[str, Any]],
        tasks: List[InferenceTask],
        request_type: Type[I],
        logging_context: LoggingContext,
    ) -> List[I]:
        requests = []
        for raw_request, task in zip(raw_requests, tasks):
            request = self._parse_input(
                raw_request, task, request_type, logging_context
            )
            if request is not None:
                requests.append(request)
        return requests


class BaseBentoServiceWithResponse(
    BaseBentoService, Generic[RE], metaclass=abc.ABCMeta
):
    @property
    @abc.abstractmethod
    def response_type(self) -> Type[RE]:
        pass

    def _format_response(
        self, response: Optional[RE], meta: Optional[Meta]
    ) -> Dict[str, Any]:
        if meta:
            seldon_message_response = SeldonMessage[self.response_type](  # type: ignore[name-defined]
                status=Status(),
                meta=meta,
                jsonData=response,
            )
        else:
            seldon_message_response = SeldonMessage[self.response_type](  # type: ignore[name-defined]
                status=Status(),
                jsonData=response,
            )
        return seldon_message_response.dict(exclude_none=True)

    def _format_responses(
        self, responses: List[RE], metas: List[Meta]
    ) -> List[Dict[str, Any]]:
        return [
            self._format_response(response, meta)
            for response, meta in zip(responses, metas)
        ]


class BasePredictor(
    BaseBentoServiceWithResponse[RE], Generic[RT, RE], metaclass=abc.ABCMeta
):
    @property
    @abc.abstractmethod
    def request_type(self) -> Type[RT]:
        pass

    @property
    def cache(self) -> Cache[RT, RE]:
        if not hasattr(self, "_cache"):
            self._cache = Cache[RT, RE](
                self,
                self.request_type,
                self.response_type,
                self.settings.redis_url,
                datetime.timedelta(seconds=self.settings.cache_duration),
            )
        return self._cache

    def _send_feedback(
        self,
        request: Optional[SeldonMessage[RT]],
        response: Optional[SeldonMessage[RE]],
        truth: Optional[SeldonMessage[RE]],
        reward: Optional[float],
        routing: Optional[int],
    ) -> None:
        if reward is not None:
            self._monitor.observe_reward(reward)

    @api(route="send-feedback", input=SeldonJsonInput(), batch=False)
    def send_feedback(
        self, raw_feedback: Dict[str, Any], task: InferenceTask = None
    ) -> Optional[Dict[str, Any]]:
        logging_context = self.get_logger_context(endpoint="send-feedback")
        logger.debug("/send-feedback: %s", raw_feedback, extra=logging_context)

        if task is None:
            task = InferenceTask()

        with self.monitor.count_exceptions(endpoint="send_feedback"), ExceptionHandler(
            [task], logging_context
        ):
            feedback = self._parse_input(
                raw_feedback,
                task,
                Feedback[self.request_type, self.response_type],  # type: ignore[name-defined]
                logging_context,
            )

            if feedback is not None:
                logger.debug("Parsed feedback: %s", feedback, extra=logging_context)

                if feedback.truth is not None and feedback.truth.meta.puid is not None:
                    logger.debug(
                        "Truth received and contains a puid. Looking for request and response in the cache...",
                        extra=logging_context,
                    )
                    cache_value = self.cache.get_cache_value(feedback.truth.meta.puid)
                    logger.debug(
                        "Cache value for puid %s: %s",
                        feedback.truth.meta.puid,
                        cache_value,
                        extra=logging_context,
                    )
                    if cache_value is not None:
                        feedback.request = SeldonMessage(
                            jsonData=cache_value.request, meta=cache_value.meta
                        )
                        feedback.response = SeldonMessage(
                            jsonData=cache_value.response, meta=cache_value.meta
                        )
                        logger.debug(
                            "Feedback reconstructed from cache value: %s",
                            feedback,
                            extra=logging_context,
                        )

                routing = (
                    feedback.response.meta.routing.get(PRED_UNIT_ID)
                    if feedback.response and feedback.response.meta
                    else None
                )

                self._send_feedback(
                    feedback.request,
                    feedback.response,
                    feedback.truth,
                    feedback.reward,
                    routing,
                )

                return self._format_response(None, None)
            return None


class BaseBatchPredictor(BasePredictor[RT, RE], Generic[RT, RE], metaclass=abc.ABCMeta):
    def _process_with_cache(
        self,
        seldon_message_requests: List[SeldonMessageRequest[RT]],
        process_fn: Callable[[List[RT]], List[RE]],
    ) -> List[RE]:
        requests = [
            seldon_message_request.jsonData
            for seldon_message_request in seldon_message_requests
        ]
        responses = self.cache.get_responses(
            [
                seldon_message_request.meta.puid
                for seldon_message_request in seldon_message_requests
            ],
            requests,
        )
        uncached_requests: List[RT] = [
            request
            for request, response in zip(requests, responses)
            if response is None
        ]
        uncached_metas = [
            seldon_message_request.meta
            for seldon_message_request, response in zip(
                seldon_message_requests, responses
            )
            if response is None
        ]

        if uncached_requests:
            uncached_responses = process_fn(uncached_requests)
            self.cache.set_responses(
                uncached_requests, uncached_responses, uncached_metas
            )

            for i, response in enumerate(responses):
                if response is None:
                    responses[i] = uncached_responses.pop(0)

        processed_responses = cast(
            List[RE], responses
        )  # all the None values have been processed
        return processed_responses

    @abc.abstractmethod
    def _predict(self, requests: List[RT]) -> List[RE]:
        pass

    @api(
        input=SeldonJsonInput(),
        batch=True,
        mb_max_latency=1000,
        mb_max_batch_size=100,
    )
    def predict(
        self,
        raw_requests: List[Dict[str, Any]],
        tasks: Optional[List[InferenceTask]] = None,
    ) -> List[Dict[str, Any]]:
        logging_context = self.get_logger_context(endpoint="predict")
        logger.debug("/predict: %s", raw_requests, extra=logging_context)
        if tasks is None:
            tasks = [InferenceTask()] * len(raw_requests)

        with self.monitor.count_exceptions(endpoint="predict"), ExceptionHandler(
            tasks, logging_context
        ):
            seldon_message_requests = self._parse_inputs(
                raw_requests,
                tasks,
                SeldonMessageRequest[self.request_type],  # type: ignore[name-defined]
                logging_context,
            )

            responses = self._process_with_cache(seldon_message_requests, self._predict)

            return self._format_responses(
                responses,
                [
                    seldon_message_request.meta
                    for seldon_message_request in seldon_message_requests
                ],
            )


class BaseSinglePredictor(
    BasePredictor[RT, RE], Generic[RT, RE], metaclass=abc.ABCMeta
):
    @abc.abstractmethod
    def _predict(self, request: RT) -> RE:
        pass

    @api(input=SeldonJsonInput(), batch=False)
    def predict(
        self, raw_request: Dict[str, Any], task: InferenceTask = None
    ) -> Optional[Dict[str, Any]]:
        logging_context = self.get_logger_context(endpoint="predict")
        logger.debug("/predict: %s", raw_request, extra=logging_context)
        if task is None:
            task = InferenceTask()

        with self.monitor.count_exceptions(endpoint="predict"), ExceptionHandler(
            [task], logging_context
        ):
            seldon_message_request = self._parse_input(
                raw_request,
                task,
                SeldonMessageRequest[self.request_type],  # type: ignore[name-defined]
                logging_context,
            )

            if seldon_message_request is not None:
                request: RT = seldon_message_request.jsonData

                logger.debug("Checking cache for %s", request, extra=logging_context)
                response = self.cache.get_response(
                    seldon_message_request.meta.puid, request
                )

                if response is None:
                    logger.debug(
                        "No cache found, predicting for %s",
                        request,
                        extra=logging_context,
                    )
                    response = self._predict(request)
                    self.cache.set_response(
                        request, response, seldon_message_request.meta
                    )

                return self._format_response(response, seldon_message_request.meta)
            return None


class BaseCombiner(
    BaseBentoServiceWithResponse[RE], Generic[RE], metaclass=abc.ABCMeta
):
    @abc.abstractmethod
    def _combine(
        self, seldon_message_list: List[SeldonMessageRequest[RE]]
    ) -> SeldonMessage[RE]:
        pass

    def _merge_meta(self, metas: List[Meta]) -> Meta:
        tags = {}
        for meta in metas:
            if meta:
                tags.update(meta.tags)
        return Meta(puid=metas[0].puid, tags=tags)

    @api(input=SeldonJsonInput(), batch=False)
    def aggregate(
        self,
        raw_seldon_message_list: List[Dict[str, Any]],
        task: InferenceTask = None,
    ) -> Optional[Dict[str, Any]]:
        logging_context = self.get_logger_context(endpoint="aggregate")
        logger.debug("/aggregate: %s", raw_seldon_message_list, extra=logging_context)
        if task is None:
            task = InferenceTask()

        with self.monitor.count_exceptions(endpoint="combine"), ExceptionHandler(
            [task], logging_context
        ):
            seldon_message_list = self._parse_input(
                raw_seldon_message_list,
                task,
                List[SeldonMessageRequest[self.response_type]],  # type: ignore[name-defined]
                logging_context,
            )

            if seldon_message_list:
                response = self._combine(seldon_message_list)

                return self._format_response(
                    response.jsonData,
                    self._merge_meta(
                        [seldon_message.meta for seldon_message in seldon_message_list]
                        + [response.meta]
                    ),
                )
            return None


class BaseRouter(BaseBentoService, Generic[RT], metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def request_type(self) -> Type[RT]:
        pass

    @property
    def cache(self) -> Cache[RT, int]:
        if not hasattr(self, "_cache"):
            self._cache = Cache[RT, int](
                self,
                self.request_type,
                int,
                self.settings.redis_url,
                datetime.timedelta(seconds=self.settings.cache_duration),
            )
        return self._cache

    @abc.abstractmethod
    def _route(self, seldon_message: SeldonMessageRequest[RT]) -> int:
        pass

    @api(input=SeldonJsonInput(), batch=False)
    def route(
        self, raw_request: Dict[str, Any], task: InferenceTask = None
    ) -> Optional[Dict[str, Any]]:
        logging_context = self.get_logger_context(endpoint="route")
        logger.debug("/route: %s", raw_request, extra=logging_context)
        if task is None:
            task = InferenceTask()

        with self.monitor.count_exceptions(endpoint="route"), ExceptionHandler(
            [task], logging_context
        ):
            seldon_message_request = self._parse_input(
                raw_request,
                task,
                SeldonMessageRequest[self.request_type],  # type: ignore[name-defined]
                logging_context,
            )

            if seldon_message_request is not None:
                request: RT = seldon_message_request.jsonData

                logger.debug("Checking cache for %s", request, extra=logging_context)
                option = self.cache.get_response(
                    seldon_message_request.meta.puid, request
                )

                if option is None:
                    logger.debug(
                        "No cache found, predicting for %s",
                        request,
                        extra=logging_context,
                    )
                    option = self._route(seldon_message_request)
                    self.cache.set_response(
                        request, option, seldon_message_request.meta
                    )

                return RoutingSeldonMessage(
                    status=Status(),
                    meta=seldon_message_request.meta,
                    data=DefaultData(tensor=Tensor(shape=[1, 1], values=[option])),
                ).dict(exclude_none=True)
            return None
