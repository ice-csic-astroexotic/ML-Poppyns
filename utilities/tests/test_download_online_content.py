"""
Test for the download_online_content.py script.

    Authors:

        Michele Ronchi (ronchi @ ice.csic.es)

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

import os
from tempfile import NamedTemporaryFile

import pytest

import utilities.download_online_content as do


@pytest.fixture()
def test_case_1():
    data = {
        "download_url_successful": "http://example.com/file.txt",
        "download_url_failure": "http://example.com/invalid_url",
        "file_expected": "Mock file content",
    }

    return data


@pytest.fixture
def temp_download_file():
    """
    Fixture to create a temporary download file.
    """
    with NamedTemporaryFile(delete=False) as temp:
        temp_file_path = temp.name
    yield temp_file_path
    os.unlink(temp_file_path)


def test_download_file_successful(
    monkeypatch, temp_download_file, test_case_1
):
    """
    Test if the download is successful.
    """
    # Define a mock function for urllib.request.urlretrieve.
    def mock_urlretrieve(url, filename):
        # Simulate successful download by creating a dummy file.
        with open(filename, "w") as f:
            f.write("Mock file content")

    # Apply the mock function using monkeypatch.
    monkeypatch.setattr("urllib.request.urlretrieve", mock_urlretrieve)

    # Call the function.
    do.download_file(
        test_case_1["download_url_successful"], temp_download_file
    )

    # Assert that the file was downloaded correctly.
    with open(temp_download_file, "r") as f:
        assert f.read() == test_case_1["file_expected"]


def test_download_file_failure(
    monkeypatch, temp_download_file, test_case_1, caplog
):
    """
    Test when then download fails.
    """
    # Define a mock function for urllib.request.urlretrieve that raises an exception.
    def mock_urlretrieve(url, filename):
        raise Exception("Download failed")

    # Apply the mock function using monkeypatch.
    monkeypatch.setattr("urllib.request.urlretrieve", mock_urlretrieve)

    # Call the function.
    do.download_file(test_case_1["download_url_failure"], temp_download_file)

    # Assert that the error message is logged.
    assert "Download failed" in caplog.text
