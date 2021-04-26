import numpy
import numpy as np
import scipy as sp
import scipy.integrate as integrate
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
    def butter_lowpass(cutOff, fs, order=4):
        nyq = 0.5 * fs
        normalCutoff = cutOff / nyq

        b, a = butter(order, normalCutoff, btype='low', analog=True)
        return b, a

    def butter_lowpass_filter(data, cutOff, fs, order=4):
        b, a = butter_lowpass(cutOff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    VitesseRG=butter_lowpass_filter(VitesseRG1, cutOff, fs, order=4)
    VitesseRD = butter_lowpass_filter(VitesseRD1, cutOff, fs, order=4)
    accRD = numpy.diff(VitesseRD1 / 3.6) * FrHZ
    accRG=numpy.diff(VitesseRG1/3.6)*FrHZ
    Vitesse_mean=(VitesseRD1+VitesseRG1)/2
    accmean = numpy.diff(Vitesse_mean / 3.6) * FrHZ
    meandist=(integrate.cumtrapz(Vitesse_mean/3.6)/FrHZ)
    distRD = (integrate.cumtrapz(VitesseRD1/3.6)/FrHZ)
    distRG = (integrate.cumtrapz(VitesseRG1/3.6)/FrHZ)


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
    for E in range(len(Events)-1):
        Vitesse_E = []
        Vitesse_E = Vitesse_mean[Events[E]:Events[E+1]]
        meandist_E = (integrate.cumtrapz(Vitesse_E/3.6)/FrHZ)
        END_Events = numpy.where(meandist_E >= 19.999)[0][0]
        Vitesse_mean_sprint = Vitesse_E[0:END_Events+1]
        VitesseRG_sprint = VitesseRG1[Events[E]:END_Events + Events[E] + 1]
        VitesseRD_sprint = VitesseRD1[Events[E]:END_Events + Events[E] + 1]

        Temps_sprint = ((END_Events+Events[E]) - Events[E])/FrHZ
        Vmean = numpy.mean(Vitesse_mean_sprint)
        Vmax = numpy.max(Vitesse_mean_sprint)
        Vmax_pos= numpy.where(Vitesse_mean_sprint==Vmax)
        Dist_vmax=meandist_E[Vmax_pos]

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


        maxpeakRD1 = VitesseRD_sprint[peaksRD[0]]
        maxpeakRD2 = VitesseRD_sprint[peaksRD[1]]
        maxpeakRD3 = VitesseRD_sprint[peaksRD[2]]

        maxpeakRG1 = VitesseRG_sprint[peaksRG[0]]
        maxpeakRG2 = VitesseRG_sprint[peaksRG[1]]
        maxpeakRG3 = VitesseRG_sprint[peaksRG[2]]

        Vmoy_startRD = (maxpeakRD1 + maxpeakRD2 + maxpeakRD3) / 3
        Vmoy_startRG = (maxpeakRG1 + maxpeakRG2 + maxpeakRG3) / 3


        return 0


