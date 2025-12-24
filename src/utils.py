import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

def obtener_mascara(sample_id, df):
    """Genera la máscara completa sumando los RLEs."""
    img_rles = df[df['ImageId'] == sample_id]['EncodedPixels'].tolist()
    mask_full = np.zeros((768, 768))
    for rle in img_rles:
        # Solo sumamos si el RLE no es NaN (en caso de imágenes sin barcos)
        if isinstance(rle, str):
            mask_full += rle_decode(rle)
    return mask_full

def visualizar_muestra(sample_id, df, path_train):
    """
    Función principal de visualización. 
    Recibe el ID, el DataFrame y la ruta de las imágenes.
    """
    # 1. Cargar imagen física
    img_path = os.path.join(path_train, sample_id)
    img_pixel = Image.open(img_path)

    # 2. Llamar al método de máscara pasando el df que recibimos
    mask_full = obtener_mascara(sample_id, df)
    
    # 3. Visualización (Tu código original)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(img_pixel)
    ax[0].set_title(f"Input Image: {sample_id}")
    ax[0].axis('off')
    
    ax[1].imshow(mask_full, cmap='gray')
    ax[1].set_title("True Mask (Ground Truth)")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def post_process_prediction(prediction, threshold=0.5):
    """
    Convierte la salida de la red en una máscara binaria.
    """
    return (prediction > threshold).astype(np.uint8)
