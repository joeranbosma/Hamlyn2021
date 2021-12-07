# coding=utf-8

"""
Module that implements functions to download
the data from the website.
"""

import os
import wget
from zipfile import ZipFile

class Downloader:
    """
    Base class with downloading utils.
    """
    def downloadFile(self, url, location=""):
        """
        Download file and with a custom progress bar.
        :param url: str, url to data.
        :param location: str, path to save.
        """
        wget.download(url, out=location)

    def unzipFile(self, pathZip, pathOut=None):
        """
        Unzip files from downloaded pathZip to pathOut.
        If pathOut is None, extracts to cwd.
        :param pathZip: str, path to zip.
        :param pathOut: str, path to save dir.
        """
        with ZipFile(pathZip, "r") as zipObj:
            zipObj.extractall(pathOut)

class RandomDownloader(Downloader):
    """
    Downloader child class that downloads the different datasets
    from the random data class.
    """
    def downloadRandomInputs(self, location=""):
        """
        Downloads the input random data.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/input_random_views.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "input_random_views.zip"))

    def downloadRandomTranslations(self, location=""):
        """
        Downloads the random translations set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/translation_random_views.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "translation_random_views.zip"))
    
    def downloadRandomSegmentations(self, location=""):
        """
        Downloads the random segmentation labels.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/segmentation_random_views.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "segmentation_random_views.zip"))

    def downloadRandomDepth(self, location=""):
        """
        Downloads the depth random set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/depth_random_views.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "depth_random_views.zip"))
    
    def downloadRandomCoords(self, location=""):
        """
        Downloads the 3D random coordinate set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/3Dcoordinates_random_views.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "3Dcoordinates_random_views.zip"))

class SequenceDownloader(Downloader):
    """
    Downloader child class that downloads the different datasets
    from the sequence data class.
    """
    def downloadSequenceInputs(self, location=""):
        """
        Downloads the input sequence data.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/input_sequences.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "input_sequences.zip"))

    def downloadSequenceTranslations(self, location=""):
        """
        Downloads the sequence translations set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/translation_sequences.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "translation_sequences.zip"))
    
    def downloadSequenceSegmentations(self, location=""):
        """
        Downloads the sequence segmentation labels.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/segmentation_sequences.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "segmentation_sequences.zip"))

    def downloadSequenceDepth(self, location=""):
        """
        Downloads the depth sequence set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/depth_sequences.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "depth_sequences.zip"))
    
    def downloadSequenceCoords(self, location=""):
        """
        Downloads the 3D sequence coordinate set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/3Dcoordinates_sequences.zip"
        self.downloadFile(url, location)
        self.unzipFile(os.path.join(location, "3Dcoordinates_sequences.zip"))

