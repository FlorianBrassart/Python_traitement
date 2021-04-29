import numpy
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from scipy import signal
from scipy.signal import find_peaks
from check_data import check_data
import matplotlib.pyplot as plt


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
    VitesseRD1 = (Mwz - (Mwxy * tan0 * signe)) * rayon_roue * 3.6;
    VitesseRG1 = (Mwz1 - (Mwxy1 * tan0 * (-signe))) * rayon_roue * 3.6;
    from scipy.signal import butter, lfilter
    from scipy.signal import freqs
    cutOff=8
    fs=FrHZ
    b, a = signal.butter(4, 14/FrHZ)
    VitesseRD = signal.filtfilt(b, a, VitesseRD1)
    VitesseRG = signal.filtfilt(b, a, VitesseRG1)

    accRD = numpy.diff(VitesseRD / 3.6) * FrHZ
    accRG=numpy.diff(VitesseRG/3.6)*FrHZ
    Vitesse_mean=(VitesseRD+VitesseRG)/2
    accmean = numpy.diff(Vitesse_mean / 3.6) * FrHZ
    meandist=(integrate.cumtrapz(Vitesse_mean/3.6)/FrHZ)
    distRD = (integrate.cumtrapz(VitesseRD/3.6)/FrHZ)
    distRG = (integrate.cumtrapz(VitesseRG/3.6)/FrHZ)


    seuil=8
    start1= numpy.where(Vitesse_mean>seuil)
    x=start1[0][0]
    while Vitesse_mean[x] > 0.21:
        x = x-1

    x=int(x)


    Event=[]
    Event_pos=[]
    Events=[]

    for e in range(len(Vitesse_mean)-x):

        if Vitesse_mean[x+e]>seuil:
           Event.append(Vitesse_mean[x+e])
           Event_pos.append(x+e)
    count=0
    Events=[x]
    for b in range(len(Event)-1):
        if Event_pos[b+1]-Event_pos[b] > 7*FrHZ:
            pos = Event_pos[b+1]
            while Vitesse_mean[pos] > 0.21:
                pos = pos - 1

            Events.append(pos)
    Events.append(len(Vitesse_mean))
    RECAP_Sprint=[]
    for E in range(len(Events)-1):
        Vitesse_E = []
        Vitesse_E = Vitesse_mean[Events[E]:Events[E+1]]
        meandist_E = (integrate.cumtrapz(Vitesse_E/3.6)/FrHZ)
        END_Events = numpy.where(meandist_E >= 19.999)[0][0]
        Vitesse_mean_sprint = Vitesse_E[0:END_Events+1]
        VitesseRG_sprint = VitesseRG[Events[E]:END_Events + Events[E] + 1]
        VitesseRD_sprint = VitesseRD[Events[E]:END_Events + Events[E] + 1]
        Acc_sprint = accmean [Events[E]:END_Events + Events[E] + 1]
        Temps_sprint = ((END_Events+Events[E]) - Events[E])/FrHZ
        Vmean = numpy.mean(Vitesse_mean_sprint)
        Vmax = numpy.max(Vitesse_mean_sprint)
        Vmax_pos= numpy.where(Vitesse_mean_sprint==Vmax)
        Dist_vmax=meandist_E[Vmax_pos]
        Accmax = np.max(Acc_sprint)
        Acc_moy = np.mean(Acc_sprint)
        peaksRG, properties = find_peaks(VitesseRG_sprint, distance=45, prominence=0.8)
        peaksRG_min=[0]
        peaksRD_min=[0]
        for pm in range(len(peaksRG)):
            if pm < len(peaksRG)-1:
                minp = numpy.min(VitesseRG_sprint[peaksRG[pm]:peaksRG[pm+1]])
                minpos=np.asarray(numpy.where(VitesseRG_sprint==minp))
            else:
                minp = numpy.min(VitesseRG_sprint[peaksRG[pm]:len(VitesseRG_sprint)])
                minpos = np.asarray(numpy.where(VitesseRG_sprint == minp))
            peaksRG_min.append(int(minpos))


        peaksRD, properties = find_peaks(VitesseRD_sprint, distance=45, prominence=0.8)
        for pm2 in range(len(peaksRD)):
            if pm2<len(peaksRD)-1:
                minp = numpy.min(VitesseRD_sprint[peaksRD[pm2]:peaksRD[pm2+1]])
                minpos=np.asarray(numpy.where(VitesseRD_sprint==minp))
            else:
                minp = numpy.min(VitesseRD_sprint[peaksRD[pm2]:len(VitesseRD_sprint)])
                minpos = np.asarray(numpy.where(VitesseRD_sprint == minp))
            peaksRD_min.append(int(minpos))
        Pos_peakmax3 = int((peaksRD[2]+peaksRG[2])/2)
        maxpeakRD1 = VitesseRD_sprint[peaksRD[0]]
        maxpeakRD2 = VitesseRD_sprint[peaksRD[1]]
        maxpeakRD3 = VitesseRD_sprint[peaksRD[2]]

        maxpeakRG1 = VitesseRG_sprint[peaksRG[0]]
        maxpeakRG2 = VitesseRG_sprint[peaksRG[1]]
        maxpeakRG3 = VitesseRG_sprint[peaksRG[2]]

        maxpeakmean1 = (maxpeakRD1 + maxpeakRG1) / 2
        maxpeakmean2 = (maxpeakRD2 + maxpeakRG2) / 2
        maxpeakmean3 = (maxpeakRD3 + maxpeakRG3) / 2

        Vmoy_startRD = (maxpeakRD1 + maxpeakRD2 + maxpeakRD3) / 3
        Vmoy_startRG = (maxpeakRG1 + maxpeakRG2 + maxpeakRG3) / 3
        Vmoy_start   = (maxpeakmean1 + maxpeakmean2 + maxpeakmean3) / 3
        accstart     = np.mean(Acc_sprint[0:Pos_peakmax3-1])

        Vmoy_stabRD = (VitesseRD_sprint[peaksRD[-5]] + VitesseRD_sprint[peaksRD[-4]] + VitesseRD_sprint[peaksRD[-3]] +
                       VitesseRD_sprint[peaksRD[-2]] + VitesseRD_sprint[peaksRD[-1]])/5
        Vmoy_stabRG = (VitesseRG_sprint[peaksRG[-5]] + VitesseRG_sprint[peaksRG[-4]] + VitesseRG_sprint[peaksRG[-3]] +
                       VitesseRG_sprint[peaksRG[-2]] + VitesseRG_sprint[peaksRG[-1]]) / 5
        Vmoy_stab   = (Vmoy_stabRD+Vmoy_stabRG)/2

        Accstab = np.mean(Acc_sprint[peaksRG[-5]:-1])

        ecart=len(peaksRD) - len(peaksRG)
        if ecart <= 0:
            sizepeak = len(peaksRD)
        else:
            sizepeak = len(peaksRG)

        tpsPRD = []
        tpsPRG = []
        tpsRRD = []
        tpsRRG = []
        tpsCRD = []
        tpsCRG = []
        tpsP = []
        tpsR = []
        tpsC = []
        Asy = []
        for c in range(sizepeak):
            tpsPRD1 = (peaksRD[c]-peaksRD_min[c])/FrHZ
            tpsPRD.append(tpsPRD1)
            tpsPRG1 = (peaksRG[c] - peaksRG_min[c]) / FrHZ
            tpsPRG.append(tpsPRG1)

            tpsP1 = (tpsPRD1 + tpsPRG1) / 2
            tpsP.append(tpsP1)

            tpsRRD1 = (peaksRD_min[c+1] - peaksRD[c]) / FrHZ
            tpsRRD.append(tpsRRD1)
            tpsRRG1 = (peaksRG_min[c+1] - peaksRG[c]) / FrHZ
            tpsRRG.append(tpsRRG1)
            tpsR1 = (tpsPRD1 + tpsPRG1) / 2
            tpsR.append(tpsR1)
            tpsCRD1 = (peaksRD_min[c+1] - peaksRD_min[c]) / FrHZ
            tpsCRD.append(tpsCRD1)
            tpsCRG1 = (peaksRG_min[c+1] - peaksRG_min[c]) / FrHZ
            tpsCRG.append(tpsCRG1)
            tpsC1 = (tpsCRD1 + tpsCRG1) / 2
            tpsC.append(tpsC1)
            if peaksRD[c] - peaksRG[c] >= 0:
                Asy1 = (peaksRD[c] - peaksRG[c]) / peaksRD[c]
                Asy.append(Asy1)
            else:
                Asy1 = (peaksRG[c] - peaksRD[c]) / peaksRG[c]
                Asy.append(Asy1)

        cadenceRD = (sizepeak - 1) / (peaksRD[-1] / FrHZ) * 60
        cadenceRG = (sizepeak - 1) / (peaksRG[-1] / FrHZ) * 60
        cadence   = (cadenceRD + cadenceRG) / 2

        RECAP_Sprint1 = [maxpeakRD1, maxpeakRG1, maxpeakmean1,
                        maxpeakRD2, maxpeakRG2, maxpeakmean2,
                        maxpeakRD3, maxpeakRG3, maxpeakmean3,
                        Vmoy_stabRD, Vmoy_stabRG, Vmoy_stab,
                        Vmoy_start, Vmax, Vmean,
                        accstart, Accmax, Acc_moy, Accstab,
                        Dist_vmax, Vmoy_start,
                        tpsP[0], tpsP[1], tpsP[2], np.mean(tpsP[0:2]), np.mean(tpsP[-5:-1]),
                        tpsR[0], tpsR[1], tpsR[2], np.mean(tpsR[0:2]), np.mean(tpsR[-5:-1]),
                        tpsC[0], tpsC[1], tpsC[2], np.mean(tpsC[0:2]), np.mean(tpsC[-5:-1]),
                        Asy[0], Asy[1], Asy[2], np.mean(Asy[0:2]), np.mean(Asy[-5:-1]), cadence
                        ]


        RECAP_Sprint.extend(RECAP_Sprint1)


    return 0


