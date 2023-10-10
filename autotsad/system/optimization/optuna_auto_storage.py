from __future__ import annotations

import logging
import socket
import tempfile
import time
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import ContextManager, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from docker import DockerClient
    from docker.models.containers import Container
    from optuna.storages import BaseStorage

from ...config import AutoTSADConfig

OPTUNA_DASHBOARD_IMAGE_NAME = "ghcr.io/optuna/optuna-dashboard:v0.10.3"
POSTGRESQL_IMAGE_NAME = "postgres:latest"
DASHBOARD_CONTAINER_NAME = "autotsad-optuna-dashboard"
DB_CONTAINER_NAME = "autotsad-optuna-db"
DB_MAX_CONNECTIONS = 1000
DB_STARTUP_DELAY = 2  # seconds
DB_CREATION_DELAY = 5  # in seconds
JOURNAL_FILENAME = "autotsad-optuna-journal.log"
SQLITE_FILENAME = "autotsad-optuna-sqlite.db"


class OptunaStorageType(Enum):
    POSTGRES = "postgres"
    SQLITE = "sqlite"
    JOURNAL = "journal"

    @staticmethod
    def from_string(s: str) -> OptunaStorageType:
        return OptunaStorageType(s.lower())


class OptunaStorageReference(ContextManager["BaseStorage"]):
    def __init__(self, tmp_path: Path = Path(tempfile.gettempdir()), storage_type: OptunaStorageType = OptunaStorageType.JOURNAL):
        self.path = tmp_path
        self.storage_type = storage_type
        self.storage: Optional[BaseStorage] = None

    @staticmethod
    def from_config(config: AutoTSADConfig) -> OptunaStorageReference:
        return OptunaStorageReference(
            tmp_path=config.general.cache_dir(),
            storage_type=OptunaStorageType.from_string(config.optimization.optuna_storage_type),
        )

    def __enter__(self) -> BaseStorage:
        self.start()
        return self.storage

    def __exit__(self, __exc_type: Type[BaseException] | None, __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        self.stop()
        return False

    def start(self) -> None:
        if self.storage_type == OptunaStorageType.POSTGRES:
            from optuna.storages import RDBStorage

            # TODO: allow remote worker to connect as well (needs public hostname)
            hostname = socket.gethostname()
            storage = RDBStorage(
                url=f"postgresql://postgres:postgres@{hostname}:5432/postgres",
                engine_kwargs={"pool_size": 1, "max_overflow": 2, "pool_timeout": 60},
            )
            self.storage = storage

        elif self.storage_type == OptunaStorageType.SQLITE:
            from optuna.storages import RDBStorage

            storage = RDBStorage(f"sqlite:///{self.path.resolve() / SQLITE_FILENAME}")
            self.storage = storage
        else:
            from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock

            journal_file_path = str(self.path / JOURNAL_FILENAME)
            storage = JournalStorage(
                JournalFileStorage(journal_file_path, lock_obj=JournalFileOpenLock(journal_file_path))
            )
            self.storage = storage

    def stop(self) -> None:
        if self.storage is not None:
            self.storage.remove_session()
            self.storage = None


class OptunaStorageManager(ContextManager):
    def __init__(self,
                 tmp_path: Path = Path(tempfile.gettempdir()),
                 storage_type: OptunaStorageType = OptunaStorageType.JOURNAL,
                 cleanup: bool = True,
                 dashboard: bool = False):
        self.cleanup: bool = cleanup
        self.storage_type = storage_type
        self.dashboard: bool = dashboard and self.storage_type == OptunaStorageType.POSTGRES
        self._log = logging.getLogger(f"autotsad.optimization.{self.__class__.__name__}")
        self._postgres_container: Optional[Container] = None
        self._dashboard_container: Optional[Container] = None
        self._docker: Optional[DockerClient] = None
        if self.storage_type == OptunaStorageType.POSTGRES or dashboard:
            import docker

            self._docker = docker.from_env()
        self._path: Path = tmp_path
        self._storage_ref: OptunaStorageReference = OptunaStorageReference(tmp_path, self.storage_type)

    @staticmethod
    def from_config(config: AutoTSADConfig) -> OptunaStorageManager:
        return OptunaStorageManager(
            tmp_path=config.general.cache_dir(),
            storage_type=OptunaStorageType.from_string(config.optimization.optuna_storage_type),
            cleanup=config.optimization.optuna_storage_cleanup,
            dashboard=config.optimization.optuna_dashboard,
        )

    def __enter__(self) -> OptunaStorageManager:
        self.start()
        return self

    def __exit__(self, __exc_type: Type[BaseException] | None, __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        self.stop()
        return False

    def _start_optuna_postgres_storage(self) -> None:
        """Create a Postgres storage for Optuna.

        Automatically starts a Postgres container if it is not already running.
        """
        from docker.errors import NotFound

        try:
            c = self._docker.containers.get(container_id=DB_CONTAINER_NAME)
            if c.status == "exited":
                # start container
                c.start()
                time.sleep(DB_STARTUP_DELAY)
                c.reload()

            if c.status != "running":
                raise RuntimeError(f"Postgres container is in unexpected state: {c.status}")
            else:
                self._postgres_container = c
        except NotFound:
            # create container:
            self._postgres_container = self._docker.containers.run(
                POSTGRESQL_IMAGE_NAME,
                f"-c max_connections={DB_MAX_CONNECTIONS}",
                name=DB_CONTAINER_NAME,
                environment={
                    "POSTGRES_PASSWORD": "postgres",
                },
                ports={"5432/tcp": "5432"},
                detach=True,
            )
            time.sleep(DB_CREATION_DELAY)
        try:
            self._log.debug(f"Postgres is running at {self._storage_ref.storage.url}")  # type: ignore
        except AttributeError:
            pass

    def _start_optuna_dashboard(self) -> None:
        from docker.errors import NotFound

        try:
            c = self._docker.containers.get(container_id=DASHBOARD_CONTAINER_NAME)
            if c.status == "exited":
                # start container
                c.start()
                time.sleep(1)
                c.reload()

            if c.status != "running":
                self._log.warning(f"Dashboard container is in an unexpected state: {c.status}!")
            else:
                self._dashboard_container = c

        except NotFound:
            try:
                assert self._storage_ref.storage is not None
                storage_url = self._storage_ref.storage.url  # type: ignore
            except (AttributeError, AssertionError) as e:
                self._log.error("Could not find dashboard connection URL to storage; not starting dashboard!",
                                exc_info=e)
                return

            # create container
            hostname = socket.gethostname()
            print(f"\n\tStarting Optuna dashboard at http://{hostname}:8080\n")
            self._log.debug(f"Optuna dashboard online at http://{hostname}:8080")
            self._dashboard_container = self._docker.containers.run(
                OPTUNA_DASHBOARD_IMAGE_NAME,
                storage_url,
                name=DASHBOARD_CONTAINER_NAME,
                network_mode="host",
                detach=True,
            )

    def get(self) -> BaseStorage:
        if self._storage_ref.storage is None:
            self.start()
        return self._storage_ref.storage

    def start(self) -> BaseStorage:
        if self.storage_type == OptunaStorageType.POSTGRES:
            self._start_optuna_postgres_storage()
        self._storage_ref.start()

        if self.dashboard:
            self._start_optuna_dashboard()

        return self._storage_ref.storage

    def stop(self) -> None:
        self._storage_ref.stop()
        time.sleep(1)

        if self._postgres_container is not None:
            self._postgres_container.stop()
            if self.cleanup:
                self._postgres_container.remove(v=True, force=True)
            self._postgres_container = None
        if self._dashboard_container is not None:
            self._dashboard_container.stop()
            if self.cleanup:
                self._dashboard_container.remove(v=True, force=True)
            self._dashboard_container = None

        if self.cleanup and (self._path / JOURNAL_FILENAME).exists():
            (self._path / JOURNAL_FILENAME).unlink()
