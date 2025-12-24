import os
os.environ["SM_FRAMEWORK"] = "tf.keras" # Obligatorio para versiones nuevas de TF
import segmentation_models as sm

def get_unet_resnet50(input_shape=(768, 768, 3)):
    """Crea una U-Net con backbone ResNet50."""
    model = sm.Unet(
        'resnet50', 
        encoder_weights='imagenet', # Pesos pre-entrenados
        classes=1, 
        activation='sigmoid', 
        input_shape=input_shape
    )
    return model