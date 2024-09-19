import argparse
import concurrent
import dataclasses
import hashlib
import math
import os
import random
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from typing import Literal, Tuple

import matplotlib
import numpy as np
import randimage
from numpy.typing import NBitBase

import log
from benchmark import benchmark
from log import Level, LogQueue, LogWriter, get_logger, init_logger

type Image = np.ndarray[Tuple[int, int, Literal[3]], np.dtype[np.floating[NBitBase]]]


@dataclass
class GenerationOptions:
    width: int
    height: int
    count: int = 1
    date_from: datetime = datetime.fromtimestamp(0)
    date_to: datetime = datetime.now()


class ImageGenerator:
    def __init__(
        self, log_queue: LogQueue, output_dir: Path, max_workers: int | None = None
    ) -> None:
        self._log_queue = log_queue
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            log.logger.debug(f"created output directory {output_dir}")
        if not output_dir.is_dir():
            raise NotADirectoryError(f"output directory {output_dir} does not exist")
        self._output_dir = output_dir

        self._max_workers: int = max_workers or os.cpu_count() or 1

    def generate_images(self, opts: GenerationOptions) -> None:
        with ProcessPoolExecutor() as executor:
            worker_count = self._max_workers
            tasks: list[Future[None]] = []

            for i in range(worker_count):
                worker_img_count = opts.count // worker_count + (
                    i < (opts.count % worker_count)
                )
                worker_opts = dataclasses.replace(opts, count=worker_img_count)

                task = executor.submit(
                    self._worker, self._log_queue, f"worker {i}", worker_opts
                )
                tasks.append(task)
                log.logger.debug(
                    f"assigned generation of {worker_img_count} images to worker {i}"
                )

            concurrent.futures.wait(tasks)

    def _worker(self, log_queue: LogQueue, name: str, gen_opts: GenerationOptions):
        init_logger(log_queue, name, Level.DEBUG)
        logger = get_logger()

        count_str_len = int(math.log10(gen_opts.count) // 1 + 1)

        for i in range(gen_opts.count):
            try:
                img = self._generate_random_image(gen_opts.width, gen_opts.height)
            except Exception as e:
                logger.error(f"generating image: {str(e)}")
                continue

            try:
                content_hash = self._hash_img_content(img)
                filename = f"{content_hash}.png"
                self._save_image(img, filename)
            except Exception as e:
                logger.error(f"saving image: {str(e)}")
                continue

            file = self._output_dir / filename
            try:
                self._set_random_date_in_range(
                    file, gen_opts.date_from, gen_opts.date_to
                )
            except Exception as e:
                logger.error(f"setting image update date: {str(e)}")
                continue

            logger.debug(
                f"{f"[{i} / {gen_opts.count}]": >{count_str_len * 2 + 5}} generated {filename}]"
            )

    def _generate_random_image(self, w: int, h: int) -> Image:
        return randimage.get_random_image((w, h))

    def _hash_img_content(self, img: Image) -> str:
        return hashlib.sha256(img.data).hexdigest()

    def _save_image(self, img: Image, name: str) -> None:
        matplotlib.image.imsave(self._output_dir / name, img)

    def _set_random_date_in_range(
        self, file: Path, date_from: datetime, date_to: datetime
    ) -> None:
        date = _get_random_dt_in_range(date_from, date_to)
        _set_file_content_update_time(file, date)


def _get_random_dt_in_range(start: datetime, end: datetime) -> datetime:
    min_date = int(start.timestamp() * 1e6)
    max_date = int(end.timestamp() * 1e6)
    ts = random.randint(min_date, max_date)
    return datetime.fromtimestamp(ts / 1e6)


def _set_file_content_update_time(path: Path, upd_time: datetime) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")
    if not path.is_file():
        raise FileNotFoundError(f"File {path} is not a file")

    atime = os.stat(path).st_atime_ns
    mtime = int(upd_time.timestamp() * 1e9)
    os.utime(path, ns=(atime, mtime))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Image Generator")
    parser.add_argument("directory", help="Directory to save images to", type=Path)
    parser.add_argument(
        "--width", dest="width", help="Image width", type=int, required=True
    )
    parser.add_argument(
        "--height", dest="height", help="Image height", type=int, required=True
    )
    parser.add_argument(
        "-c", "--count", dest="count", help="Image count", type=int, default=1
    )
    parser.add_argument(
        "--date-from",
        dest="date_from",
        help="Random image modification date from",
        type=datetime.fromisoformat,
    )
    parser.add_argument(
        "--date-to",
        dest="date_to",
        help="Random image modification date to",
        type=datetime.fromisoformat,
    )
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

    w = args.width
    h = args.height
    count = args.count

    gen = ImageGenerator(log_queue, args.directory, max_workers=4)
    opts = GenerationOptions(
        width=w,
        height=h,
        count=count,
        date_from=args.date_from or GenerationOptions.date_from,
        date_to=args.date_to or GenerationOptions.date_to,
    )

    log.logger.info(
        f"starting generation of {count} {w}x{h} images in {args.directory}"
    )
    t = benchmark(gen.generate_images, opts)
    log.logger.info(f"completed in {round(t / 1e6, 5)} ms")

    log_queue.put_nowait(None)
    log_writer.stop()


if __name__ == "__main__":
    main()
