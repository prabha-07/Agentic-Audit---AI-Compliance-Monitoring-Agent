from __future__ import annotations

from rq import Connection, Worker

from backend.jobs.queue import redis_conn


def main() -> None:
    conn = redis_conn()
    with Connection(conn):
        worker = Worker(["analysis"])
        worker.work()


if __name__ == "__main__":
    main()
