from typing import Generic, List, Optional, Type, TypeVar, Union

import datetime
import logging
from hashlib import sha256

from bentoml import BentoService
from pydantic import BaseModel
from pydantic.generics import GenericModel

from bento2seldon.seldon import DEPLOYMENT_ID, PRED_UNIT_ID, PRED_UNIT_KEY, Meta

RT = TypeVar("RT", bound=BaseModel)
RE = TypeVar("RE", bound=Union[BaseModel, int])


logger = logging.getLogger(__name__)


class CacheValue(GenericModel, Generic[RT, RE]):
    request: RT
    response: RE
    meta: Meta


class Cache(Generic[RT, RE]):
    def __init__(
        self,
        bento_service: BentoService,
        request_type: Type[RT],
        response_type: Type[RE],
        redis_url: Optional[str],
        expiration_delta: datetime.timedelta,
    ) -> None:
        self._name: str = bento_service.name
        self._version: str = bento_service.version
        self._request_type = request_type
        self._response_type = response_type
        try:
            from redis import Redis

            self._redis = (
                Redis.from_url(redis_url, decode_responses=True)
                if redis_url is not None
                else None
            )
        except:
            logger.warning("redis not installed.")
            self._redis = None

        self._expiration_delta = expiration_delta

    def _request_to_key(self, request: RT) -> str:
        request_hash = sha256(request.json().encode("utf-8")).hexdigest()
        return self._request_hash_to_key(request_hash)

    def _request_hash_to_key(self, request_hash: str) -> str:
        return f"{self._name}:{DEPLOYMENT_ID}:{self._version}:request:{request_hash}"

    def _wrap_puid(self, puid: str) -> str:
        return f"{self._name}:{DEPLOYMENT_ID}:{self._version}:puid:{puid}"

    def should_cache(self, request: RT, response: RE, meta: Meta) -> bool:
        logger.debug("Verifying if should cache: %s, %s, %s", request, response, meta)
        return meta.tags.get(PRED_UNIT_KEY) == PRED_UNIT_ID

    def set_response(self, request: RT, response: RE, meta: Meta) -> None:
        if self._redis:
            if self.should_cache(request, response, meta):
                key = self._request_to_key(request)
                value = CacheValue[RT, RE](
                    request=request, response=response, meta=meta
                )
                puid = self._wrap_puid(meta.puid)

                logger.debug("Caching %s=%s=%s", puid, key, value)

                self._redis.set(puid, key, ex=self._expiration_delta)
                self._redis.set(key, value.json(), ex=self._expiration_delta)
        else:
            logger.warning("Redis not available.")

    def get_response(self, puid: str, request: RT) -> Optional[RE]:
        if self._redis:
            key = self._request_to_key(request)
            value = self._redis.get(key)
            response: Optional[RE] = (
                CacheValue[self._request_type, self._response_type]  # type: ignore[name-defined]
                .parse_raw(value)
                .response
                if value
                else None
            )
            if response is not None:
                puid = self._wrap_puid(puid)

                logger.debug("Caching %s=%s", puid, key)

                self._redis.set(puid, key, ex=self._expiration_delta)
            return response
        else:
            logger.warning("Redis not available.")
            return None

    def set_responses(
        self, requests: List[RT], responses: List[RE], metas: List[Meta]
    ) -> None:
        if self._redis:
            puids, keys, values = zip(
                *(
                    (
                        self._wrap_puid(meta.puid),
                        self._request_to_key(request),
                        CacheValue(
                            request=request, response=response, meta=meta
                        ).json(),
                    )
                    for request, response, meta in zip(requests, responses, metas)
                    if self.should_cache(request, response, meta)
                )
            )

            logger.debug("Caching multiple values: %s=%s=%s", puids, keys, values)

            self._redis.mset({**dict(zip(puids, keys)), **dict(zip(keys, values))})

            for k in keys + puids:
                self._redis.expire(k, self._expiration_delta)
        else:
            logger.warning("Redis not available.")

    def get_responses(self, puids: List[str], requests: List[RT]) -> List[Optional[RE]]:
        if self._redis:
            keys = [self._request_to_key(request) for request in requests]
            responses = [
                CacheValue[self._request_type, self._response_type]  # type: ignore[name-defined]
                .parse_raw(value)
                .response
                if value
                else None
                for value in self._redis.mget(keys)
            ]
            puids = [self._wrap_puid(puid) for puid in puids]

            logger.debug("Caching %s=%s", puids, keys)

            puid_to_key_mapping = {
                puid: key
                for puid, key, response in zip(puids, keys, responses)
                if response is not None
            }

            if puid_to_key_mapping:
                self._redis.mset(
                    {
                        puid: key
                        for puid, key, response in zip(puids, keys, responses)
                        if response is not None
                    }
                )
                for puid in puid_to_key_mapping.keys():
                    self._redis.expire(puid, self._expiration_delta)
            return responses
        else:
            logger.warning("Redis not available.")
            return [None] * len(requests)

    def get_cache_value(self, puid: str) -> Optional[CacheValue[RT, RE]]:
        if self._redis:
            puid = self._wrap_puid(puid)
            key = self._redis.get(puid)

            logger.debug("Getting cache value for %s", puid)
            if key:
                logger.debug(
                    "The key for %s exists and is %s. Getting the value...",
                    key,
                    puid,
                )
                value = self._redis.get(key)
                if value:
                    cache_value = CacheValue[
                        self._request_type, self._response_type  # type: ignore[name-defined]
                    ].parse_raw(value)
                    logger.debug("Found cache value for %s: %s", puid, value)
                    return cache_value
                return None
        else:
            logger.warning("Redis not available.")
        return None

    def get_all(self) -> List[CacheValue[RT, RE]]:
        if self._redis:
            return [
                CacheValue.parse_raw(value)
                for value in self._redis.mget(
                    self._redis.keys(self._request_hash_to_key("*"))
                )
                if value is not None
            ]
        else:
            logger.warning("Redis not available.")
            return []
