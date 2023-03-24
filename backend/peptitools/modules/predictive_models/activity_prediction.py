"""Activity Prediction module"""
import multiprocessing as mp
from random import random

import pandas as pd
from joblib import load

from peptitools.modules.machine_learning_tools.numerical_representation.run_encoding import Encoding
from peptitools.modules.utils import ConfigTool
from peptitools.modules.machine_learning_tools.numerical_representation.physicochemical_properties import Physicochemical
from peptitools.modules.machine_learning_tools.numerical_representation.fft_encoding import FftEncoding


class ActivityPrediction(ConfigTool):
    """Activity Prediction Class"""

    def __init__(self, data, options, is_file, config):
        super().__init__("activity_prediction", data, config, is_file)
        self.options = options
        self.config = config
        self.encoder_dataset = pd.read_csv(config["folders"]["encoders_dataset"], index_col=0)
        self.models_folder = config["folders"]["activity_prediction_models"]
        self.dataset_encoded_path = f"{self.static_folder}/{round(random() * 10**20)}.csv"
        self.options = options
        self.full_dataset_encoded = pd.DataFrame()
        self.cores = mp.cpu_count()
        self.activities_list = pd.read_csv(config["folders"]["activity_table"])

    def __process_encoding_stage(self):
        """Encode sequences using selected method"""
        with open(self.temp_file_path, "r", encoding="utf-8") as file:
            self.data = self.create_df(file.read())
        for group in self.encoder_dataset.columns:
            physicochemical = Physicochemical(self.data, group, self.config["folders"]["encoders_dataset"], "id", "sequence")
            self.dataset_encoded = physicochemical.encode_dataset()
            fft = FftEncoding(self.dataset_encoded , "id", vector_size=150)
            self.dataset_encoded = fft.encoding_dataset()
            self.dataset_encoded["group"] = int(group[-1]) + 1
            self.full_dataset_encoded = pd.concat([self.full_dataset_encoded, self.dataset_encoded], axis = 0)

    def __load_model(self, idactivity, idgroup):
        """Load model using joblib"""
        model_path = f"{self.models_folder}/{idactivity}/{idgroup}"
        return load(model_path)

    def __evaluate_activity(self, idactivity):
        """Evaluate activity"""
        act_row = self.activities_list[self.activities_list["idactivity"] == idactivity]
        name = act_row.name.values[0]
        data = []
        for _, row in self.full_dataset_encoded.iterrows():
            group = row["group"]
            model = self.__load_model(idactivity, group)
            values = row.tolist()[1:-1]
            data.append(
                {
                    "idpeptide": row["id"].split(" ")[0],
                    "group": group,
                    "idactivity": idactivity,
                    "activity": name,
                    "prediction": model.predict([values])[0],
                }
            )
        evaluation = pd.DataFrame(data)
        grouped = evaluation.groupby(
            ["idpeptide", "idactivity", "activity"], as_index=False
        ).sum()
        grouped["probability"] = grouped["prediction"] / self.encoder_dataset.shape[0]
        grouped.drop(["group", "prediction"], axis=1, inplace=True)
        return grouped

    def evaluate(self):
        """Evaluate all specified activities"""
        df_evaluation = pd.concat(
            [
                self.__evaluate_activity(idactivity)
                for idactivity in self.options["activities"]
            ]
        )
        response = []
        for idpeptide in df_evaluation.idpeptide.unique():
            sub_df = df_evaluation[df_evaluation["idpeptide"] == idpeptide][
                ["activity", "probability"]
            ]
            response.append(
                {
                    "id": idpeptide,
                    "data": sub_df.values.tolist(),
                    "columns": [
                        " ".join(a.capitalize().split("_"))
                        for a in sub_df.columns.tolist()
                    ],
                }
            )
        return response

    def run_process(self):
        """Run all activity prediction process"""
        self.__process_encoding_stage()
        return self.evaluate()
