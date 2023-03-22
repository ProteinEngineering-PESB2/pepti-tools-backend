"""Encoding module"""
import os
import subprocess
from random import random
from peptitools.modules.utils import fasta2df
from peptitools.modules.machine_learning_tools.numerical_representation.one_hot_encoding import OneHotEncoding
from peptitools.modules.machine_learning_tools.numerical_representation.physicochemical_properties import Physicochemical
from peptitools.modules.machine_learning_tools.numerical_representation.fft_encoding import FftEncoding
from peptitools.modules.utils import ConfigTool


class Encoding(ConfigTool):
    """Encoding class"""

    def __init__(self, data, options, is_file, config):
        super().__init__("encoding", data, config, is_file)
        self.options = options
        self.df = fasta2df(data)
        self.encoder_dataset = config["folders"]["encoders_dataset"]
        random_number = str(round(random() * 10**20))
        self.output_path = f"results/{random_number}.csv"

    def process_encoding(self):
        """Encoding process"""
        print(self.options)
        if self.options["encoding"] == "one_hot_encoding":
            one_hot_encoding = OneHotEncoding(self.df, "id", "sequence")
            res = one_hot_encoding.encode_dataset()
        if self.options["encoding"] in ["physicochemical_properties", "digital_signal_processing"]:
            physicochemical = Physicochemical(self.df, self.options["selected_property"], self.encoder_dataset, "id", "sequence")
            res = physicochemical.encode_dataset()
            print(res)
        if self.options["encoding"] == "digital_signal_processing":
            fft = FftEncoding(res, "id")
            res = fft.encoding_dataset()
        self.dataset_encoded = res
        self.save_csv_on_static(res, self.output_path)
        return {"path": self.output_path}