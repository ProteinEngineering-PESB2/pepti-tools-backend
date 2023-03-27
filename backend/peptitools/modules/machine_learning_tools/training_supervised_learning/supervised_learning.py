"""Supervised learning module"""
import pandas as pd
from joblib import dump
from peptitools.modules.machine_learning_tools.transformer.transformation_data import Transformer
from peptitools.modules.machine_learning_tools.numerical_representation.run_encoding import Encoding
from peptitools.modules.machine_learning_tools.training_supervised_learning.run_algorithm import RunAlgorithm
from peptitools.modules.utils import ConfigTool


class SupervisedLearning(Encoding):
    """Supervised Learning class"""

    def __init__(self, data, options, is_file, config, is_fasta):
        super().__init__(data, options, is_file, config, is_fasta)
        self.options = options
        self.task = self.options["task"]
        self.algorithm = self.options["algorithm"]
        self.validation = self.options["validation"]
        self.test_size = self.options["test_size"]
        self.kernel = self.options["kernel"]
        self.preprocessing = self.options["preprocessing"]
        self.transformer = Transformer()

        self.target = self.data.target
        self.data.drop("target", inplace=True, axis=1)
        
        self.model = None

    def run(self):
        """Runs encoding, preprocessing and build ML model"""
        self.process_encoding()
        self.dataset_encoded.drop(["id"], axis=1, inplace=True)
        run_instance = RunAlgorithm(
            self.dataset_encoded,
            self.target,
            self.task,
            self.algorithm,
            self.validation,
            self.test_size,
        )
        response_training = run_instance.training_model()
        if self.test_size != 0:
            response_testing = run_instance.testing_model()
            if self.task == "regression":
                temp = response_testing["performance"]
                response_testing["performance_testing"] = temp
                del response_testing["performance"]

                temp = response_testing["corr"]
                response_testing["corr_testing"] = temp
                del response_testing["corr"]

                temp = response_testing["scatter"]
                response_testing["scatter_testing"] = temp
                del response_testing["scatter"]

                temp = response_testing["error_values"]
                response_testing["error_values_testing"] = temp
                del response_testing["error_values"]

            elif self.task == "classification":
                temp = response_testing["performance"]
                response_testing["performance_testing"] = temp
                del response_testing["performance"]

                temp = response_testing["confusion_matrix"]
                response_testing["confusion_matrix_testing"] = temp
                del response_testing["confusion_matrix"]

                temp = response_testing["analysis"]
                response_testing["analysis_testing"] = temp
                del response_testing["analysis"]

            response_training.update(response_testing)
        self.model = run_instance.get_model()
        self.job_path = self.output_path.replace(".csv", ".joblib")
        self.dump_joblib()
        return response_training

    def dump_joblib(self):
        """Save model"""
        dump(self.model, self.job_path)
