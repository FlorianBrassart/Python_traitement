# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:58:54 2021

@author: flo_b
"""
print('démarrage du programme')
import json

from Sprint_repetes_IMU import Traitement_sprint_IMU
from os import listdir
folder = listdir("DATAS")
fichier_roue_droite = [s for s in folder if s.__contains__("Droite")]
fichier_roue_gauche = [s for s in folder if s.__contains__("Gauche")]
fichier_info = [s for s in folder if s.__contains__("information")]
with open("DATAS/" + fichier_roue_gauche[0], "r") as read_file:
    DATARG = json.load(read_file)
with open("DATAS/" + fichier_roue_droite[0], "r") as read_file:
    DATARD = json.load(read_file)
with open("DATAS/" + fichier_info[0], "r") as read_file:
    INFOS = json.load(read_file)


RECAP_Sprint_best, RECAP_Sprint_last = Traitement_sprint_IMU(DATARG, DATARD, INFOS)
print('fin des calculs')
from PDF_Prossessing import pdf_prossessing
pdf = pdf_prossessing(RECAP_Sprint_best, RECAP_Sprint_last, INFOS)
print('fin du traitement')

