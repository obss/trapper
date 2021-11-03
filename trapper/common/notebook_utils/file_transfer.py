import os
import urllib.request
from pathlib import Path


def download_from_url(from_url: str, to_path: str):

    Path(to_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(to_path):
        urllib.request.urlretrieve(
            from_url,
            to_path,
        )
