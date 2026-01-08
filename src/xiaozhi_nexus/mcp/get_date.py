import datetime
from typing import Tuple


def get_date() -> Tuple[str, str]:
    time_zone = "Asia/Shanghai"
    # 可选的时间戳，格式为ISO 8601，例如：2023-10-01T12:00:00Z
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return time_zone, timestamp
