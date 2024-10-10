"""
    This script downloads a file from a given URL into the specified directory path.

    Authors:

        Michele Ronchi (ronchi@ice.csic.es)
"""

import logging
import sys
import urllib.request

log = logging.getLogger(__name__)


def download_file(download_url: str, destination_path: str) -> None:
    """
    Download a file from a given URL into the specified directory path.

    Args:
        download_url (str): Download URL for the file.
        destination_path (str): Path where to save the downloaded file; the file name has to be included.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    log = logging.getLogger(__name__)

    try:
        # Download the file using urllib.request.urlretrieve.
        urllib.request.urlretrieve(download_url, destination_path)
        log.info("Download completed successfully.")
    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
