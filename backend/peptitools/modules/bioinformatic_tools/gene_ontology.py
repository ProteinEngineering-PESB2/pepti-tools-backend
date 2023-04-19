"""Gene ontology module"""
import subprocess
import os
from random import random
import pandas as pd

class GeneOntology:
    """Gene ontology class"""
    def __init__(self, fasta_path, config, options):
        self.fasta_path = fasta_path
        self.config = config
        self.results_folder = config["folders"]["results_folder"]
        self.output_path = os.path.realpath(f"{self.results_folder}/{round(random() * 10**20)}.aln")
        self.molecular_function = options["molecular_function"]
        self.biological_process = options["biological_process"]
        self.celular_component = options["celular_component"]
        self.ontologies = self.parse_ontologies()

    def parse_ontologies(self):
        """Returns a string with options for metastudent"""
        ontologies = []
        if self.molecular_function:
            ontologies.append("MFO")
        if self.biological_process:
            ontologies.append("BPO")
        if self.celular_component:
            ontologies.append("CCO")
        return ",".join(ontologies)

    def run_process(self):
        """Execute metastudent and return results"""
        command = [
            "metastudent",
            "-i",
            self.fasta_path,
            "-o",
            self.output_path,
            "--ontologies",
            self.ontologies,
        ]
        subprocess.check_output(command)
        response = self.find_and_load_data()

        if len(response) == 0:
            return {
                "status": "warning",
                "description": "There's no significant results for this sequences"
            }
        return response

    def find_and_load_data(self):
        """Parse metastudent results and returns json"""
        results = []
        params = [
            (self.molecular_function, ".MFO.txt", "molecular_function"),
            (self.biological_process, ".BPO.txt", "biological_process"),
            (self.celular_component, ".CCO.txt", "celular_component"),
        ]
        for param_tuple in params:
            if param_tuple[0]:
                try:
                    df_go = pd.read_csv(
                        self.output_path + param_tuple[1], header=None, sep="\t"
                    )
                    df_go.rename(
                        columns={0: "id_seq", 1: "id_go", 2: "probability", 3: "term"},
                        inplace=True,
                    )
                    unique_go = df_go.id_seq.unique()
                    go_array = []
                    for single_go in unique_go:
                        temp = df_go[df_go.id_seq == single_go][
                            ["id_go", "probability", "term"]
                        ]
                        go_array.append(
                            {"id_seq": single_go, "results": temp.to_dict("records")}
                        )
                    results.append({"type": param_tuple[2], "prediction": go_array})
                except:
                    print("No result for molecular function")
        return results
