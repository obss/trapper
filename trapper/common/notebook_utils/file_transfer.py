import os
import urllib.request
from pathlib import Path
from typing import Optional


def download_from_url(url: str, to_path: Optional[str] = None):
    """

    Args:
        url:
        to_path:

    Returns:

    """
    if to_path is None:
        to_path = os.path.basename(url)

    Path(to_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(to_path):
        urllib.request.urlretrieve(
            url,
            to_path,
        )
