"""
This script downloads a file from a given URL into the specified directory path.

Authors:

    Michele Ronchi (ronchi@ice.csic.es)

Copyright (c) MAGNESIA (ICE-CSIC) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import sys

import requests

log = logging.getLogger(__name__)


def download_file(download_url: str, destination_path: str) -> None:
    """
    Download a file from a given URL into the specified directory path.

    Args:
        download_url (str): download URL for the file.
        destination_path (str): path where to save the downloaded file; the file name has to be included.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    try:
        # Send an HTTP GET request to the URL to download the file.
        with requests.get(download_url, stream=True) as response:
            # Check if the request was successful (status code 200).
            if response.status_code == 200:
                # Open a file handle to write the response content to.
                with open(destination_path, "wb") as f:
                    # Iterate over the response content in chunks and write each chunk to the file.
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                log.info("Download completed successfully.")
            else:
                log.info(
                    f"Failed to download the file. Status code: {response.status_code}"
                )
    except Exception as e:
        log.error(f"An error occurred: {str(e)}")
