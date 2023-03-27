"""Bioinformatic tools routes"""
import configparser

from flask import Blueprint, request

from peptitools.modules.bioinformatic_tools.gene_ontology import GeneOntology
from peptitools.modules.bioinformatic_tools.msa_module import MultipleSequenceAlignment
from peptitools.modules.bioinformatic_tools.structural_characterization import StructuralCharacterization
from peptitools.modules.bioinformatic_tools.pfam_domain import Pfam
from peptitools.modules.utils import Interface

##Reads config file and asign folder names.
config = configparser.ConfigParser()
config.read("config.ini")

bioinfo_tools_blueprint = Blueprint("bioinfo_tools_blueprint", __name__)


@bioinfo_tools_blueprint.route("/msa/", methods=["POST"])
def apply_msa():
    """Multiple sequence alignment route"""
    data, is_file = Interface(request).parse_without_options()
    msa = MultipleSequenceAlignment(data, is_file, config)
    check = msa.check
    if check["status"] == "error":
        return check
    result = msa.run_process()
    return {"result": result}


@bioinfo_tools_blueprint.route("/structural_analysis/", methods=["POST"])
def apply_structural_analysis():
    """Structural analysis route"""
    data, options, is_file = Interface(request).parse_with_options()
    data_array = [">" + a for a in data.split(">")[1:]]
    result = []
    for data in data_array:
        structural = StructuralCharacterization(data, options, is_file, config)
        check = structural.check
        if check["status"] == "error":
            return check
        result.append(structural.run_process())
    return {"result": result}


@bioinfo_tools_blueprint.route("/gene_ontology/", methods=["POST"])
def apply_gene_ontology():
    """Gene ontology route"""
    data, options, is_file = Interface(request).parse_with_options()
    go_obj = GeneOntology(data, options, is_file, config)
    check = go_obj.check
    if check["status"] == "error":
        return check
    result = go_obj.process()
    if len(result) == 0:
        return {
            "status": "warning",
            "description": "There's no significant results for this sequences",
        }
    return {"result": result}


@bioinfo_tools_blueprint.route("/pfam/", methods=["POST"])
def apply_pfam():
    """Pfam route"""
    data, is_file = Interface(request).parse_without_options()
    pf_obj = Pfam(data, is_file, config)
    check = pf_obj.check
    if check["status"] == "error":
        return check
    result = pf_obj.process()
    if len(result) == 0:
        return {
            "status": "warning",
            "description": "There's no significant results for this sequences",
        }
    return {"result": result}
