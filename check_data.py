import numpy as np


def check_data(Roue_G, Roue_D):
    Roue_G = dict(Roue_G, gyro_z= np.asarray(Roue_G["gyro_z"])* (4000/pow(2,16)), gyro_x=np.asarray(Roue_G["gyro_x"]) * (4000/pow(2,16)), gyro_y=np.asarray(Roue_G["gyro_y"]) * (4000/pow(2,16)) )
    Roue_D = dict(Roue_D, gyro_z= np.asarray(Roue_D["gyro_z"]) * (4000/pow(2,16)), gyro_x=np.asarray(Roue_D["gyro_x"]) * (4000/pow(2,16)), gyro_y=np.asarray(Roue_D["gyro_y"]) * (4000/pow(2,16)) )

    if np.mean(Roue_G["gyro_z"]) < 0:
        Roue_G = dict(Roue_G, gyro_y=(np.negative(Roue_G["gyro_y"])).tolist(), gyro_x=(np.negative(Roue_G["gyro_x"])).tolist(), gyro_z=(np.negative(Roue_G["gyro_z"])).tolist())
        print("inversion roue gauche")
    if np.mean(Roue_D["gyro_z"]) < 0:
        Roue_D = dict(Roue_D, gyro_y=(np.negative(Roue_D["gyro_y"])).tolist(), gyro_x=(np.negative(Roue_D["gyro_x"])).tolist(), gyro_z=(np.negative(Roue_D["gyro_z"])).tolist())
        print("inversion roue droite")
    if len(Roue_G["gyro_z"]) < len(Roue_D["gyro_z"]):
        Roue_D["gyro_z"] = Roue_D["gyro_z"][0:len(Roue_G["gyro_z"])]

    if len(Roue_G["gyro_z"]) > len(Roue_D["gyro_z"]):
        Roue_G["gyro_z"] = Roue_G["gyro_z"][0:len(Roue_D["gyro_z"])]

    return Roue_G, Roue_D
