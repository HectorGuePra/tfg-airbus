#!/usr/bin/env python
import os
# 1. Configuración de entorno y Framework
os.environ["SM_FRAMEWORK"] = "tf.keras" 

import matplotlib
matplotlib.use('Agg') # Obligatorio para ejecutar en servidores sin interfaz gráfica
import matplotlib.pyplot as plt

import sys
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LambdaCallback

# Añadimos la ruta para encontrar tus archivos en la carpeta src
sys.path.append(os.path.abspath(".."))
from src.utils import rle_decode 
from src.model import get_unet_resnet50

import shutil

# Borrar la carpeta de resultados si ya existe para empezar de cero
if os.path.exists('output_images'):
    shutil.rmtree('output_images')
    print("Carpeta 'output_images' borrada para nuevo entrenamiento.")

# --- CONFIGURACIÓN DE GPU ---
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    print(f"Usando GPU: {physical_devices[0]}")

# --- CONFIGURACIÓN DE RUTAS ---
PATH_CSV = '../data/train_ship_segmentations_v2.csv'
PATH_TRAIN = '../data/train_v2/'

# --- PREPARACIÓN DE DATOS ---
df = pd.read_csv(PATH_CSV)

# Filtramos imágenes con barcos para combatir el desbalance de clases
df_with_ships = df.dropna(subset=['EncodedPixels'])
ship_table = df_with_ships.groupby('ImageId').size().reset_index(name='num_ships')
ship_table = ship_table.sort_values(by='num_ships', ascending=False).reset_index(drop=True)

print(f"Total de imágenes con barcos cargadas: {len(ship_table)}")

# --- PREPROCESAMIENTO Y GENERADOR ---
# Obtenemos la función de preprocesamiento específica para ResNet50
preprocess_input = sm.get_preprocessing('resnet50')

def data_generator(dataframe, batch_size=4):
    while True:
        batch_df = dataframe.sample(batch_size)
        images, masks = [], []
        
        for img_id in batch_df['ImageId']:
            # Carga y conversión a RGB
            img = cv2.imread(os.path.join(PATH_TRAIN, img_id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Aplicamos el preprocesamiento de la red (normalización ImageNet)
            img = preprocess_input(img).astype('float32')
            
            # Generar máscara combinada (float32 evita errores de tipos)
            all_rles = df[df['ImageId'] == img_id]['EncodedPixels'].values
            mask = np.zeros((768, 768, 1), dtype='float32')
            for rle in all_rles:
                if isinstance(rle, str):
                    mask[:, :, 0] += rle_decode(rle)
            
            images.append(img)
            masks.append(np.clip(mask, 0, 1).astype('float32'))
            
        yield np.array(images, dtype='float32'), np.array(masks, dtype='float32')

# --- FUNCIONES DE GUARDADO ---
def guardar_predicciones(modelo_ia, generador, epoca, num=3):
    if not os.path.exists('output_images'):
        os.makedirs('output_images')
        
    img_batch, mask_batch = next(generador)
    preds = modelo_ia.predict(img_batch)
    
    plt.figure(figsize=(15, 5 * num))
    for i in range(min(num, len(img_batch))):
        plt.subplot(num, 3, i*3 + 1); plt.imshow(img_batch[i]); plt.title("Original")
        plt.subplot(num, 3, i*3 + 2); plt.imshow(mask_batch[i,:,:,0], cmap='gray'); plt.title("Real")
        # Visualizamos la probabilidad con un umbral de 0.5
        plt.subplot(num, 3, i*3 + 3); plt.imshow(preds[i,:,:,0] > 0.5, cmap='gray'); plt.title("Predicción")
    
    plt.savefig(f'output_images/prediccion_epoca_{epoca}.png')
    plt.close()

# --- MODELO Y ENTRENAMIENTO ---
# La U-Net usa ResNet50 como encoder preentrenado
model = get_unet_resnet50()

# Compilación con pérdida híbrida ideal para objetos pequeños
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=sm.losses.bce_dice_loss,
    metrics=[sm.metrics.iou_score]
)

# Callbacks para automatizar el servidor
log_csv = CSVLogger('history_log.csv', append=False)
checkpoint = ModelCheckpoint('mejor_modelo.h5', monitor='iou_score', mode='max', save_best_only=True)
img_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: guardar_predicciones(model, train_gen, epoch))

# Entrenamiento real
train_gen = data_generator(ship_table, batch_size=4)

print("Iniciando entrenamiento en segundo plano...")
model.fit(
    train_gen, 
    steps_per_epoch=300, 
    epochs=40, 
    callbacks=[log_csv, checkpoint, img_callback]
)

print("Entrenamiento completado. Revisa 'history_log.csv' y la carpeta 'output_images'.")