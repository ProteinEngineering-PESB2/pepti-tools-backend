"""Machine learning routes"""
import configparser

from flask import Blueprint, request

from peptitools.modules.machine_learning_tools.clustering_methods.alignment_clustering import AlignmentClustering
from peptitools.modules.machine_learning_tools.clustering_methods.clustering_process import Clustering
from peptitools.modules.machine_learning_tools.clustering_methods.distance_clustering import DistanceClustering
from peptitools.modules.machine_learning_tools.numerical_representation.run_encoding import Encoding

from peptitools.modules.machine_learning_tools.transformer.pca_process import PCA
from peptitools.modules.machine_learning_tools.training_supervised_learning.supervised_learning import SupervisedLearning
from peptitools.modules.utils import Interface

##Reads config file and asign folder names.
config = configparser.ConfigParser()
config.read("config.ini")


machine_learning_blueprint = Blueprint("machine_learning_blueprint", __name__)


@machine_learning_blueprint.route("/encoding/", methods=["POST"])
def apply_encoding():
    """Encode a fasta file or text"""
    data, options, is_file = Interface(request).parse_with_options()
    code = Encoding(data, options, is_file, config)
    check = code.check
    if check["status"] == "error":
        return check
    result = code.process_encoding()
    return {"result": result}

@machine_learning_blueprint.route("/clustering/", methods=["POST"])
def api_clustering():
    """It performs clustering from a fasta file or text"""
    data, options, is_file = Interface(request).parse_with_options()
    clustering_object = Clustering(data, options, is_file, config)
    check = clustering_object.check
    if check["status"] == "error":
        return check
    result = clustering_object.process_clustering()
    return {"result": result}


@machine_learning_blueprint.route("/alignment_clustering/", methods=["POST"])
def api_alignment_clustering():
    """It performs clustering from a fasta file or text"""
    data, is_file = Interface(request).parse_without_options()
    clustering_object = AlignmentClustering(data, is_file, config)
    check = clustering_object.check
    if check["status"] == "error":
        return check
    result = clustering_object.run_clustering()
    return {"result": result}


@machine_learning_blueprint.route("/distance_clustering/", methods=["POST"])
def api_distance_clustering():
    """It performs clustering from a fasta file or text"""
    data, options, is_file = Interface(request).parse_with_options()
    clustering_object = DistanceClustering(data, options, is_file, config)
    check = clustering_object.check
    if check["status"] == "error":
        return check
    result = clustering_object.run_process()
    return {"result": result}

@machine_learning_blueprint.route("/pca/", methods=["POST"])
def api_pca():
    """It performs a PCA from a stored dataframe"""
    pca = PCA(request.json["params"], config["folders"]["static_folder"])
    result, path = pca.apply_pca()
    return {"result": result, "path": path}

@machine_learning_blueprint.route("/supervised_learning/", methods=["POST"])
def api_supervised_learning():
    """It performs a Supervised learning from a csv file"""
    data, options, is_file = Interface(request).parse_with_options()
    sl_obj = SupervisedLearning(data, options, is_file, config)
    check = sl_obj.check
    if check["status"] == "error":
        return check
    result = sl_obj.run()
    job_path = sl_obj.job_path
    return {"result": result, "job_path": job_path}