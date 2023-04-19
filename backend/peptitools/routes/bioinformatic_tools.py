"""Bioinformatic tools routes"""
import configparser

from flask import Blueprint, request

from peptitools.modules.bioinformatic_tools.gene_ontology import GeneOntology
from peptitools.modules.bioinformatic_tools.msa import MultipleSequenceAlignment
from peptitools.modules.bioinformatic_tools.structural_characterization import StructuralCharacterization
from peptitools.modules.bioinformatic_tools.pfam_domain import Pfam
from peptitools.modules.utils import Interface
from peptitools.modules.utils import parse_response
import json

##Reads config file and asign folder names.
config = configparser.ConfigParser()
config.read("config.ini")

bioinfo_tools_blueprint = Blueprint("bioinfo_tools_blueprint", __name__)


@bioinfo_tools_blueprint.route("/msa/", methods=["POST"])
def apply_msa():
    """Multiple sequence alignment route"""
    check = parse_response(request, config, "msa", False, "fasta")
    if check["status"] == "error":
        return check
    msa = MultipleSequenceAlignment(check["path"], config)
    result = msa.run_process()
    return {"result": result, "status": "success"}

@bioinfo_tools_blueprint.route("/pfam/", methods=["POST"])
def apply_pfam():
    """Pfam route"""
    check = parse_response(request, config, "pfam", False, "fasta")
    if check["status"] == "error":
        return check
    pfam = Pfam(check["path"], config)
    result = pfam.run_process()
    return {"result": result, "status": "success"}

@bioinfo_tools_blueprint.route("/gene_ontology/", methods=["POST"])
def apply_gene_ontology():
    """Gene ontology route"""
    check = parse_response(request, config, "gene_ontology", False, "fasta")
    if check["status"] == "error":
        return check
    go = GeneOntology(check["path"], config, json.loads(request.form["options"]))
    result = go.run_process()
    return {"result": result, "status": "success"}

@bioinfo_tools_blueprint.route("/structural_analysis/", methods=["POST"])
def apply_structural_analysis():
    """Structural analysis route"""
    check = parse_response(request, config, "structural", False, "fasta")
    if check["status"] == "error":
        return check
    struct = StructuralCharacterization(check["path"], config, json.loads(request.form["options"]))
    result = struct.run_process()
    return {"result": result, "status": "success"}

