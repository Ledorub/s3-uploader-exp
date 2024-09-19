import argparse
import os
import queue
from datetime import datetime
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Any, Protocol

import boto3

import log
from benchmark import benchmark
from log import Level, LogQueue, LogWriter, get_logger, init_logger

type FileQueue = Queue[Path | None]


class S3Client(Protocol):
    def create_bucket(self, *, Bucket: str) -> Any: ...

    def upload_file(self, file_name: Path, bucket: str, object_name: str) -> Any: ...


class S3ClientFactory(Protocol):
    def create_client(self) -> S3Client: ...


class Boto3S3ClientFactory:
    def __init__(self, url: str):
        self._url = url

    def _create_session(self) -> boto3.session.Session:
        return boto3.session.Session()

    def create_client(self) -> S3Client:
        s = self._create_session()
        return s.client("s3", endpoint_url=self._url)


class Uploader:
    def __init__(
        self,
        log_queue: LogQueue,
        client_factory: S3ClientFactory,
        max_workers: int | None = None,
    ):
        self._log_queue = log_queue
        self._client_factory = client_factory
        if max_workers is not None and max_workers < 2:
            raise ValueError(
                "max_workers cannot be less than 2: at least one walker and one uploader"
            )
        if max_workers is None:
            cpus = os.cpu_count()
            self._max_workers: int = cpus if cpus is not None and cpus > 1 else 2

        self._queue_size = 100

    def upload_folder(self, path: Path):
        file_queue: FileQueue = Queue(self._queue_size)
        log.logger.debug(f"created queue of size {self._queue_size}")

        uploader_count = self._max_workers - 1

        walker = self._start_walker(path, file_queue, uploader_count)
        log.logger.debug("started the walker")
        uploaders = self._start_uploaders(file_queue, uploader_count)
        log.logger.debug(f"started {uploader_count} uploaders")

        self._wait([walker, *uploaders])
        log.logger.debug(f"completed uploading")

    def _start_uploaders(self, file_queue: FileQueue, count: int) -> list[Process]:
        procs = []
        for i in range(count):
            name = f"uploader_{i}"
            proc = Process(
                target=self._upload,
                args=(self._log_queue, name, file_queue, self._client_factory),
            )
            procs.append(proc)
            proc.start()
            log.logger.debug(f"started the worker {name}")

        return procs

    def _upload(
        self,
        log_queue: LogQueue,
        name: str,
        file_queue: Queue,
        client_factory: S3ClientFactory,
    ):
        init_logger(log_queue, name, Level.DEBUG)
        logger = get_logger()

        client = client_factory.create_client()
        while True:
            try:
                filename = file_queue.get_nowait()
            except queue.Empty:
                continue

            if filename is None:
                break

            mod_dt = _get_update_time(filename)
            bucket_name = mod_dt.strftime("%Y-%m-%d")
            obj_name = mod_dt.strftime("%H-%M-%S") + filename.suffix

            try:
                self._create_bucket_if_not_exists(client, bucket_name)
                client.upload_file(filename, bucket_name, obj_name)
                logger.debug(f"uploaded {filename} as {obj_name} to {bucket_name}")
            except Exception as e:
                logger.error(
                    f"uploading {filename} as {obj_name} to {bucket_name}: {str(e)}"
                )

    def _create_bucket_if_not_exists(self, client: S3Client, name: str):
        try:
            client.create_bucket(Bucket=name)
        except client.exceptions.BucketAlreadyOwnedByYou:
            pass

    def _start_walker(
        self, path: Path, file_queue: Queue, sentinel_count: int
    ) -> Process:
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")
        if not path.is_dir():
            raise NotADirectoryError(f"Path {path} is not a directory")

        proc = Process(
            target=self._walk, args=(self._log_queue, path, file_queue, sentinel_count)
        )
        proc.start()
        return proc

    def _walk(
        self, log_queue: LogQueue, path: Path, file_queue: Queue, sentinel_count: int
    ) -> None:
        init_logger(log_queue, "walker", Level.DEBUG)
        logger = get_logger()

        for filename in path.iterdir():
            if filename.is_dir():
                continue
            file_queue.put(filename)
            logger.debug(f"enqueued {filename}")

        for _ in range(sentinel_count):
            file_queue.put(None)

    def _wait(self, procs: list[Process]) -> None:
        for proc in procs:
            proc.join()


def _get_update_time(path: Path) -> datetime:
    if not path.exists():
        raise ValueError(f"Path {path} does not exist")

    upd_time = path.stat().st_mtime
    return datetime.fromtimestamp(upd_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="S3 Uploader")
    parser.add_argument("directory", help="A directory to upload from", type=Path)
    parser.add_argument("--host", dest="host", help="S3 host", required=True)
    parser.add_argument(
        "--debug",
        dest="debug",
        help="Show debug output",
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_queue: LogQueue = Manager().Queue(-1)
    log_lvl = Level.DEBUG if args.debug else Level.INFO
    init_logger(log_queue, __name__, log_lvl)

    log_writer = LogWriter(log_queue, log_lvl)
    log_writer.start()

    client_factory = Boto3S3ClientFactory(args.host)
    uploader = Uploader(log_queue, client_factory)

    log.logger.info(f"starting uploading of images from {args.directory}")
    t = benchmark(uploader.upload_folder, args.directory)
    log.logger.info(f"completed uploading in {round(t / 1e6, 5)} ms")

    log_queue.put_nowait(None)
    log_writer.stop()


if __name__ == "__main__":
    main()
