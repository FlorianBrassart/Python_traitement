import numpy
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from check_data import check_data


def Traitement_sprint_IMU(Roue_G, Roue_D, INFOS):
    Roue_G, Roue_D = check_data(Roue_G, Roue_D)
    FrHZ=128
    dif=np.asarray(Roue_G["gyro_z"])-np.asarray(Roue_D["gyro_z"])
    dif[dif==0]=0.0001
    signe=dif/abs(dif)
    Mwz=np.deg2rad(np.asarray(Roue_D["gyro_z"]))
    Mwxy = np.deg2rad(np.sqrt(np.square(np.asarray(Roue_D["gyro_x"]))+np.square(np.asarray(Roue_D["gyro_y"]))))
    tan0=np.deg2rad(np.tan(float(INFOS["angle_carossage"])))
    Mwz1 = np.deg2rad(np.asarray(Roue_G["gyro_z"]))
    Mwxy1 = np.deg2rad(np.sqrt(np.square(np.asarray(Roue_G["gyro_x"])) + np.square(np.asarray(Roue_G["gyro_y"]))))
    rayon_roue=float(INFOS["taille_roues"])/78.74
    VitesseRD = (Mwz - (Mwxy * tan0 * signe)) * rayon_roue * 3.6;
    VitesseRG = (Mwz1 - (Mwxy1 * tan0 * (-signe))) * rayon_roue * 3.6;
    Vitesse_mean=(VitesseRD+VitesseRG)/2
    meandist=(integrate.cumtrapz(Vitesse_mean/3.6)/FrHZ)
    distRD = (integrate.cumtrapz(VitesseRD/3.6)/FrHZ)
    distRG = (integrate.cumtrapz(VitesseRG/3.6)/FrHZ)

    start1= numpy.where(Vitesse_mean>8)
    start1= round(start1[1]-((8/1.6)*FrHZ))
    start= numpy.where(Vitesse_mean[start1:len(Vitesse_mean)]>0.2)

    Event=0
    Events=0



    return 0


