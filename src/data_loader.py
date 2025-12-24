import numpy as np
import pandas as pd
import tensorflow as tf

#Función que convierte en imagen las mascaras de train
def rle_decode(mask_rle, shape=(768, 768)):
    if pd.isna(mask_rle): # Si no hay ningun barco se devuelve matriz de 0
        return np.zeros(shape).T #Pongo t para que coincida con el formato de kaggle
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # En el csv vienen donde empieza el barco y los pixeles que ocupa 
    starts -= 1 # Lista de donde empiezan los barcos
    ends = starts + lengths # Lista donde acaban los barcos
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T #Kaggle codifica por columnas y Python lee por filas