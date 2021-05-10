import json

import numpy as np

from PDF_Prossessing import pdf_prossessing
from Sprint_repetes_IMU import Traitement_sprint_IMU

with open("DATAS/A2_Roue Gauche_raw.json", "r") as read_file:
    DATARG = json.load(read_file)
with open("DATAS/A1_Roue Droite_raw.json", "r") as read_file:
    DATARD = json.load(read_file)
with open("DATAS/information_evaluation.json", "r") as read_file:
    INFOS = json.load(read_file)


RECAP_Sprint_best, RECAP_Sprint_last = Traitement_sprint_IMU(DATARG, DATARD, INFOS)

pdf = pdf_prossessing(RECAP_Sprint_best, RECAP_Sprint_last, INFOS)