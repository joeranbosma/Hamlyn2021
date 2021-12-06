# coding=utf-8

"""
Module that implements functions to download
the data from the website.
"""

import wget

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

    def downloadRandomTranslations(self, location=""):
        """
        Downloads the random translations set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/translation_random_views.zip"
        self.downloadFile(url, location)
    
    def downloadRandomSegmentations(self, location=""):
        """
        Downloads the random segmentation labels.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/segmentation_random_views.zip"
        self.downloadFile(url, location)

    def downloadRandomDepth(self, location=""):
        """
        Downloads the depth random set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/depth_random_views.zip"
        self.downloadFile(url, location)
    
    def downloadRandomCoords(self, location=""):
        """
        Downloads the 3D random coordinate set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/3Dcoordinates_random_views.zip"
        self.downloadFile(url, location)

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

    def downloadSequenceTranslations(self, location=""):
        """
        Downloads the sequence translations set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/translation_sequences.zip"
        self.downloadFile(url, location)
    
    def downloadSequenceSegmentations(self, location=""):
        """
        Downloads the sequence segmentation labels.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/segmentation_sequences.zip"
        self.downloadFile(url, location)

    def downloadSequenceDepth(self, location=""):
        """
        Downloads the depth sequence set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/depth_sequences.zip"
        self.downloadFile(url, location)
    
    def downloadSequenceCoords(self, location=""):
        """
        Downloads the 3D sequence coordinate set.
        """
        url = "http://opencas.dkfz.de/video-sim2real/data/3Dcoordinates_sequences.zip"
        self.downloadFile(url, location)

