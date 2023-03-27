"""Encoding module"""
from random import random
from peptitools.modules.utils import fasta2df
from peptitools.modules.machine_learning_tools.numerical_representation.one_hot_encoding import OneHotEncoding
from peptitools.modules.machine_learning_tools.numerical_representation.physicochemical_properties import Physicochemical
from peptitools.modules.machine_learning_tools.numerical_representation.fft_encoding import FftEncoding
from peptitools.modules.machine_learning_tools.numerical_representation.protein_language_model import Bioembeddings
from peptitools.modules.machine_learning_tools.numerical_representation.descriptors import Descriptor
from peptitools.modules.machine_learning_tools.transformer.transformation_data import Transformer
from peptitools.modules.utils import ConfigTool
import pandas as pd

class Encoding(ConfigTool):
    """Encoding class"""
    def __init__(self, data, options, is_file, config, is_fasta = True, api_form = "encoding"):
        super().__init__(api_form, data, config, is_file, is_fasta)
        self.options = options
        if is_fasta:
            self.data = fasta2df(data)
        else:
            self.data = pd.read_csv(self.temp_file_path)
        self.encoder_dataset = config["folders"]["encoders_dataset"]
        self.output_path = f'{config["folders"]["results_folder"]}/{round(random() * 10**20)}.csv'
        self.ids = self.data.id
        self.transformer = Transformer()


    def process_encoding(self):
        """Encoding process"""
        if self.options["encoding"] == "one_hot_encoding":
            one_hot_encoding = OneHotEncoding(self.data, "id", "sequence")
            self.dataset_encoded = one_hot_encoding.encode_dataset()
        if self.options["encoding"] in ["physicochemical_properties", "digital_signal_processing"]:
            physicochemical = Physicochemical(self.data, self.options["selected_property"], self.encoder_dataset, "id", "sequence")
            self.dataset_encoded = physicochemical.encode_dataset()
        if self.options["encoding"] == "digital_signal_processing":
            fft = FftEncoding(self.dataset_encoded , "id")
            self.dataset_encoded = fft.encoding_dataset()
        if self.options["encoding"] == "embedding":
            bio_embeddings = Bioembeddings(self.data, "id", "sequence")
            if self.options["pretrained_model"] == "bepler":
                self.dataset_encoded = bio_embeddings.apply_bepler()
            if self.options["pretrained_model"] == "fasttext":
                self.dataset_encoded = bio_embeddings.apply_fasttext()
            if self.options["pretrained_model"] == "glove":
                self.dataset_encoded = bio_embeddings.apply_glove()
            if self.options["pretrained_model"] == "plus_rnn":
                self.dataset_encoded = bio_embeddings.apply_plus_rnn()
            if self.options["pretrained_model"] == "word2vec":
                self.dataset_encoded = bio_embeddings.apply_word2vec()
        if self.options["encoding"] == "global_descriptor":
            descriptor = Descriptor(self.data, "id", "sequence")
            self.dataset_encoded = descriptor.encode_dataset()
        if "kernel" in self.options.keys() or "preprocessing" in self.options.keys():
            self.dataset_encoded = self.dataset_encoded.drop(columns=["id"])
            self.transform_data()
        
        header = ["id"] + self.dataset_encoded.columns[:-1].tolist()
        self.dataset_encoded = self.dataset_encoded[header]
        self.dataset_encoded.to_csv(self.output_path)
        return {"path": self.output_path}

    def transform_data(self):
        if self.options["kernel"] != "":
            self.dataset_encoded = self.transformer.apply_kernel_pca(self.dataset_encoded, self.options["kernel"])
        if self.options["preprocessing"] != "":
            self.dataset_encoded = self.transformer.apply_scaler(self.dataset_encoded, self.options["preprocessing"])
        if self.options["kernel"] != "" or self.options["preprocessing"] != "":
            self.dataset_encoded = pd.DataFrame(self.dataset_encoded, columns=[
                f"p_{a}" for a in range(len(self.dataset_encoded[0]))
            ])
        self.dataset_encoded["id"] = self.ids