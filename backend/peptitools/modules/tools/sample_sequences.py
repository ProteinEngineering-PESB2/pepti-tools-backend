import pandas as pd
import os
import re
import json
from peptitools.modules.utils import fasta2df, df2fasta

class SampleSequences:
    def __init__(self, sample_path, limit):
        self.sample_path = sample_path
        self.limit = int(limit)
    def get_sample(self):
        with open(self.sample_path, "r", encoding="utf-8") as file:
            text = file.read()
        data = fasta2df(text)
        data = data.sample(self.limit)
        return  {"data": df2fasta(data)}
        