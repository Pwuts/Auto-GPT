"""Redis memory provider."""
from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np
import redis
from colorama import Fore, Style
from redis.commands.search.query import Query
from redis_om import Field, JsonModel, KNNExpression, Migrator, VectorFieldOptions

from autogpt.config import Config
from autogpt.logs import logger
from autogpt.memory.context import MemoryItemRelevance

from ..memory_item import MemoryItem
from ..utils import Embedding, get_embedding
from .abstract import ContextMemoryProvider


class RedisMemory(ContextMemoryProvider):
    cfg: Config

    redis: redis.Redis
    index_name: str

    id_seq: int
    """Last sequential ID in index"""

    @property
    def id_seq_key(self):
        return f"{self.index_name}-id_seq"

    def __init__(self, cfg: Config):
        """
        Initializes the Redis memory provider.

        Args:
            cfg: The config object.

        Returns: None
        """
        self.cfg = cfg

        redis_host = cfg.redis_host
        redis_port = cfg.redis_port
        redis_password = cfg.redis_password
        redis_conn = self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=0,  # Cannot be changed
            decode_responses=True,
        )
        index_name = self.index_name = cfg.memory_index

        # Check redis connection
        try:
            self.redis.ping()
        except redis.ConnectionError as e:
            logger.typewriter_log(
                "FAILED TO CONNECT TO REDIS",
                Fore.RED,
                Style.BRIGHT + str(e) + Style.RESET_ALL,
            )
            logger.double_check(
                "Please ensure you have setup and configured Redis properly for use. "
                + f"You can check out {Fore.CYAN + Style.BRIGHT}"
                f"https://docs.agpt.co/configuration/memory/#redis-setup{Style.RESET_ALL}"
                " to ensure you've set up everything correctly."
            )
            exit(1)

        global RedisMemoryItem

        class RedisMemoryItem(JsonModel, MemoryItem):
            DIMENSION = 1536

            raw_content: str = Field(index=True, full_text_search=True)
            summary: str = Field(index=True, full_text_search=True)
            chunks: list[str]
            chunk_summaries: list[str]
            e_summary: list[float] = Field(
                index=True,
                vector_options=VectorFieldOptions.hnsw(
                    dimension=DIMENSION,
                    type=VectorFieldOptions.TYPE.FLOAT32,
                    distance_metric=VectorFieldOptions.DISTANCE_METRIC.IP,
                ),
            )
            e_chunks: list[list[float]] = Field(
                index=True,
                vector_options=VectorFieldOptions.hnsw(
                    dimension=DIMENSION,
                    type=VectorFieldOptions.TYPE.FLOAT32,
                    distance_metric=VectorFieldOptions.DISTANCE_METRIC.IP,
                ),
            )
            metadata: dict[str, bool | int | str]

            class Meta:
                database = redis_conn
                global_key_prefix = index_name

            # Necessary because JsonModel also overrides __eq__
            def __eq__(self, other):
                return MemoryItem.__eq__(self, other)

        try:
            Migrator().run()
        except Exception as e:
            logger.error(f"Error creating Redis search indexes: {e}")

        if cfg.wipe_redis_on_start:
            self.clear()

    def __iter__(self) -> Iterator[RedisMemoryItem]:
        for pk in RedisMemoryItem.all_pks():
            yield RedisMemoryItem.get(pk)  # type: ignore

    def __contains__(self, x: MemoryItem | RedisMemoryItem) -> bool:
        return RedisMemoryItem.find(RedisMemoryItem.raw_content == x.raw_content).count() > 0  # type: ignore

    def __len__(self) -> int:
        return len(list(RedisMemoryItem.all_pks()))

    def add(self, item: MemoryItem | RedisMemoryItem) -> RedisMemoryItem:
        if isinstance(item, RedisMemoryItem):
            return item.save()

        if isinstance(item.e_summary, np.ndarray):
            item.e_summary = item.e_summary.tolist()
        item.e_chunks = [
            e_chunk.tolist() if isinstance(e_chunk, np.ndarray) else e_chunk
            for e_chunk in item.e_chunks
        ]
        return RedisMemoryItem(**vars(item)).save()

    def get_relevant(self, query: str, k: int) -> Sequence[MemoryItemRelevance]:
        v_query = np.array(get_embedding(query), np.float32)

        knn_search = RedisMemoryItem.find(
            # RedisMemoryItem.raw_content % "test",
            knn=KNNExpression(
                k, RedisMemoryItem.__fields__["e_summary"], v_query.tobytes()
            )
        )
        logger.debug(f"RediSearch query for get_relevant(): {knn_search.query}")
        results: list[RedisMemoryItem] = knn_search.execute()

        return [
            MemoryItemRelevance(
                memory_item=mi,
                for_query=query,
                summary_relevance_score=1 - mi._e_summary_score,
                chunk_relevance_scores=np.dot(mi.e_chunks, v_query),
            )
            for mi in results
        ]

    def discard(self, item: MemoryItem | RedisMemoryItem):
        if not isinstance(item, RedisMemoryItem):
            matches = RedisMemoryItem.find(
                RedisMemoryItem.raw_content == item.raw_content
            ).all()
            if len(matches) > 1:
                logger.warn(
                    "Skipping ambiguous delete of memory item "
                    f"with {len(matches)} matches"
                )
                return
            if matches:
                return RedisMemoryItem.delete(matches[0].pk)
        else:
            return RedisMemoryItem.delete(item.pk)

    def clear(self) -> None:
        """Clears the Redis database."""
        logger.debug("Clearing Redis memory store")
        for key in RedisMemoryItem.all_pks():
            RedisMemoryItem.delete(key)
