import sqlite3
import numpy as np
from typing import TypeVar, Generic, MutableMapping, Iterator, Tuple
from abc import abstractmethod
import redis
import re
import os
import subprocess
import redis_server
import asyncio
import time

TK = TypeVar('TK')
TV = TypeVar('TV')


class KeyValueStorage(Generic[TK, TV], MutableMapping[TK, TV]):
    """
    A high-performance key-value storage on disk, powered by SQLite underneath.
    Generic abstract class.
    """
    def __init__(self, kvs):
        self.kvs = kvs
        self.kvs.execute("CREATE TABLE IF NOT EXISTS kv_store (key BLOB PRIMARY KEY, value BLOB)")

    @abstractmethod
    def encode_key(self, k: TK) -> bytes:
        pass

    @abstractmethod
    def decode_key(self, k: bytes) -> TK:
        pass

    @abstractmethod
    def encode_value(self, v: TV) -> bytes:
        pass

    @abstractmethod
    def decode_value(self, v: bytes) -> TV:
        pass

    def __setitem__(self, k: TK, v: TV) -> None:
        self.kvs.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            (self.encode_key(k), self.encode_value(v))
        )
        self.kvs.connection.commit()

    def __delitem__(self, k: TK) -> None:
        self.kvs.execute(
            "DELETE FROM kv_store WHERE key = ?",
            (self.encode_key(k),)
        )
        self.kvs.connection.commit()

    def __getitem__(self, k: TK) -> TV:
        cursor = self.kvs.execute(
            "SELECT value FROM kv_store WHERE key = ?",
            (self.encode_key(k),)
        )
        result = cursor.fetchone()
        if result is None:
            raise KeyError(k)
        return self.decode_value(result[0])

    def __len__(self) -> int:
        cursor = self.kvs.execute("SELECT COUNT(*) FROM kv_store")
        return cursor.fetchone()[0]

    def __iter__(self) -> Iterator[TV]:
        for k, v in self.items():
            yield v

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def items(self) -> Iterator[Tuple[TK, TV]]:
        cursor = self.kvs.execute("SELECT key, value FROM kv_store")
        for raw_key, raw_value in cursor:
            yield self.decode_key(raw_key), self.decode_value(raw_value)

    def close(self) -> None:
        self.kvs.close()


class StringVectorStorage(KeyValueStorage[str, np.ndarray]):

    def __init__(self, kvs, dtype=np.float32):
        super(StringVectorStorage, self).__init__(kvs)
        self.dtype = dtype

    def encode_key(self, k: str) -> bytes:
        return k.encode()

    def decode_key(self, k: bytes) -> str:
        return k.decode()

    def encode_value(self, v: np.ndarray) -> bytes:
        return v.astype(self.dtype).tobytes()

    def decode_value(self, v: bytes) -> np.ndarray:
        return np.frombuffer(v, dtype=self.dtype)

    @classmethod
    def open(cls, file: str, dtype=np.float32, mode: str = 'r'):
        connection = sqlite3.connect(file)
        kvs = connection.cursor()
        return cls(kvs, dtype=dtype)


class StringStringStorage(KeyValueStorage[str, str]):

    def __init__(self, kvs):
        super(StringStringStorage, self).__init__(kvs)

    def encode_key(self, k: str) -> bytes:
        return k.encode()

    def decode_key(self, k: bytes) -> str:
        return k.decode()

    def encode_value(self, k: str) -> bytes:
        return k.encode()

    def decode_value(self, k: bytes) -> str:
        return k.decode()

    @classmethod
    def open(cls, file: str, mode: str = 'r'):
        connection = sqlite3.connect(file)
        kvs = connection.cursor()
        return cls(kvs)





class KeyValueStore:

    def __init__(self, redis_client = None ):
        self.redis_client = redis_client

    def clean_ds(self):
        self.redis_client.flushdb()

    def parse_file(self, filename):
        """
        Parses a file containing key-value pairs separated by a tab.
        """
        with open(filename, 'r') as file:
            return dict(line.strip().split('\t', 1) for line in file if '\t' in line)

    def store_in_redis(self, filepath):

        # if self.redis_client.dbsize() > 0:
        #     return

        prefix = self._get_prefix_from_filepath(filepath)
        key_value_pairs = self.parse_file(filepath)

        for key, value in key_value_pairs.items():
            redis_key = f"{prefix}{key}"
            self.redis_client.set(redis_key, value)


    def get_all_keys(self):
        all_keys = self.redis_client.keys('*')

        # Convert from bytes to string if needed
        all_keys = [key.decode('utf-8') for key in all_keys]

        print(all_keys)

    def get_value(self, filepath, key):
        """
        Retrieves a value from Redis by key, using the file name as the prefix.
        """
        prefix = self._get_prefix_from_filepath(filepath)
        redis_key = f"{prefix}{key}"
        value = self.redis_client.get(redis_key)
        return value.decode('utf-8') if value else None

    def _get_prefix_from_filepath(self, filepath):
        """
        Extracts the file name from the path and returns it as a prefix with an underscore.
        """
        filename = os.path.basename(filepath)
        return f"{filename}_"

    def _get_highest_suffix(self, filepath):
        """
        Retrieves the highest numerical suffix from the keys under a given prefix.
        """
        prefix = self._get_prefix_from_filepath(filepath)
        # pattern = re.compile(rf"{prefix}(.*?)-(\d+)$")
        pattern = re.compile(rf"{re.escape(prefix)}.*-(.)(\d+)$")

        max_suffix = -1

        for key in self.redis_client.scan_iter(f"{prefix}*"):
            key_str = key.decode('utf-8')
            # print("key_str",key_str)
            match = pattern.search(key_str)
            if match:
                number = int(match.group(2))
                # print(number)
                if number > max_suffix:
                    max_suffix = number
        pre = "P" if prefix == "train.l_" else "H"
        new_key = f"SNLI-train-{pre}{max_suffix}"

        return new_key

    def get_the_next_id(self, current, filepath):

        # prefix = self._get_prefix_from_filepath(filepath)
        prefix_ = 'SNLI-train'
        # pattern = re.compile(rf"{prefix}(.*?)-(\d+)$")
        pattern = re.compile(rf"{re.escape(prefix_)}-(.)(\d+)$")

        # key_str = current.decode('utf-8')
        match = pattern.match(current)
        if match:
            number = int(match.group(2))

        new_suffix = number + 1
        pre = "P" if filepath == "l" else "H"
        new_key = f"SNLI-train-{pre}{new_suffix}"
        return new_key

    def add_new_entry(self, filepath, value):


        prefix = self._get_prefix_from_filepath(filepath)
        highest_suffix = self._get_highest_suffix(filepath)
        new_suffix = highest_suffix + 1
        new_key = f"{prefix}train-{new_suffix}"
        self.redis_client.set(new_key, value)
        return new_key


    @classmethod
    def  open(cls, redis_host='localhost', redis_port=6379, redis_db=0):

        # result = await subprocess.run([redis_server.REDIS_CLI_PATH, "ping"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # output = result.stdout.decode('utf-8').strip()


        process = subprocess.Popen([redis_server.REDIS_SERVER_PATH])
        # os.system(redis_server.REDIS_SERVER_PATH)
        # process.wait()
        time.sleep(5)

        redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)


        # print(f"all keys: {redis_client.keys('*')}")
        # redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost')
        # print(f"redis_url  {redis_url}")
        return cls(redis_client)

    def __del__(self):

        try:
            print("Shutting down Redis server...")
            self.redis_client.shutdown()
        except redis.ConnectionError:
            print("Could not connect to Redis server to shut it down.")
        except redis.exceptions.ResponseError as e:
            print(f"Error shutting down Redis: {e}")


# from bsddb3 import db
# import numpy as np
# from typing import TypeVar, Generic, MutableMapping, Iterator, Tuple
# from abc import abstractmethod
#
# TK = TypeVar('TK')
# TV = TypeVar('TV')
#
#
# class KeyValueStorage(Generic[TK, TV], MutableMapping[TK, TV]):
#     """
#     A high-performance key-value storage on disk, powered by BerkeleyDB underneath.
#     Generic abstract class.
#     """
#     def __init__(self, kvs):
#         self.kvs = kvs
#
#     @abstractmethod
#     def encode_key(self, k: TK) -> bytes:
#         pass
#
#     @abstractmethod
#     def decode_key(self, k: bytes) -> TK:
#         pass
#
#     @abstractmethod
#     def encode_value(self, v: TV) -> bytes:
#         pass
#
#     @abstractmethod
#     def decode_value(self, v: bytes) -> TV:
#         pass
#
#     def __setitem__(self, k: TK, v: TV) -> None:
#         self.kvs.put(self.encode_key(k), self.encode_value(v))
#
#     def __delitem__(self, k: TK) -> None:
#         self.kvs.delete(self.encode_key(k))
#
#     def __getitem__(self, k: TK) -> TV:
#         return self.decode_value(self.kvs.get(self.encode_key(k)))
#
#     def __len__(self) -> int:
#         return self.kvs.stat()["ndata"]
#
#     def __iter__(self) -> Iterator[TV]:
#         for k, v in self.items():
#             yield v
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.close()
#
#     def items(self) -> Iterator[Tuple[TK, TV]]:
#         cursor = self.kvs.cursor()
#         entry = cursor.first()
#         while entry:
#             raw_key, raw_value = entry
#             yield self.decode_key(raw_key), self.decode_value(raw_value)
#             entry = cursor.next()
#
#     def close(self) -> None:
#         self.kvs.close()
#
#
# class StringVectorStorage(KeyValueStorage[str, np.ndarray]):
#
#     def __init__(self, kvs, dtype=np.float32):
#         super(StringVectorStorage, self).__init__(kvs)
#         self.dtype = dtype
#
#     def encode_key(self, k: str) -> bytes:
#         return k.encode()
#
#     def decode_key(self, k: bytes) -> str:
#         return k.decode()
#
#     def encode_value(self, v: np.ndarray) -> bytes:
#         return v.astype(self.dtype).tobytes()
#
#     def decode_value(self, v: bytes) -> np.ndarray:
#         return np.frombuffer(v, dtype=self.dtype)
#
#     @classmethod
#     def open(cls, file: str, dtype=np.float32, db_kind=db.DB_BTREE, mode: str = 'r'):
#         kvs = db.DB()
#         db_mode = {
#             'r': db.DB_DIRTY_READ,
#             'w': db.DB_CREATE
#         }[mode]
#         kvs.open(file, None, db_kind, db_mode)
#         return cls(kvs, dtype=dtype)
#
#
# class StringStringStorage(KeyValueStorage[str, str]):
#
#     def __init__(self, kvs):
#         super(StringStringStorage, self).__init__(kvs)
#
#     def encode_key(self, k: str) -> bytes:
#         return k.encode()
#
#     def decode_key(self, k: bytes) -> str:
#         return k.decode()
#
#     def encode_value(self, k: str) -> bytes:
#         return k.encode()
#
#     def decode_value(self, k: bytes) -> str:
#         return k.decode()
#
#     @classmethod
#     def open(cls, file: str, db_kind=db.DB_BTREE, mode: str = 'r'):
#         kvs = db.DB()
#         db_mode = {
#             'r': db.DB_DIRTY_READ,
#             'w': db.DB_CREATE
#         }[mode]
#         kvs.open(file, None, db_kind, db_mode)
#         return cls(kvs)
