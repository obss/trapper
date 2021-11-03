import os
import urllib.request
from pathlib import Path
from typing import Optional


def download_from_url(url: str, destination: Optional[str] = None) -> None:
    """
    Utility function to download data from a specified url.

    Args:
        url: Source url of data to be downloaded.
        destination: Destination where the downloaded data is placed. If None,
            base name of the url is used, i.e if url="a/b.txt", it will be
            downloaded to "./b.txt".
    """
    if destination is None:
        destination = os.path.basename(url)

    Path(destination).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(destination):
        urllib.request.urlretrieve(
            url,
            destination,
        )
