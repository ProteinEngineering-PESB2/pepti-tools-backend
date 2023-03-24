"""Model using routes"""
import configparser

from flask import Blueprint, request

from peptitools.modules.predictive_models.activity_prediction import ActivityPrediction
from peptitools.modules.predictive_models.activity_list import ActivityList
from peptitools.modules.utils import Interface

config = configparser.ConfigParser()
config.read("config.ini")
models_blueprint = Blueprint("models_blueprint", __name__)

@models_blueprint.route("/activity_prediction/", methods=["POST"])
def apply_activity_prediction():
    """Encode a fasta file or text"""
    data, options, is_file = Interface(request).parse_with_options()
    act_pred = ActivityPrediction(data, options, is_file, config)
    check = act_pred.check
    if check["status"] == "error":
        return check
    return act_pred.run_process()


@models_blueprint.route("/activity_models_list/", methods=["GET"])
def apply_activity_models_list():
    """List all activities with predictive model"""
    act_list = ActivityList()
    return act_list.get_activity_list(config)
