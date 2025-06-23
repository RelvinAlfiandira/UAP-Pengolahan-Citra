import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):
    green = image[:, :, 1]  # channel hijau
    cropped = green[50:350, 100:700]  # sesuaikan area invisible ink

    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(cropped, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    asm = graycoprops(glcm, 'ASM')[0]         # array 4 nilai
    idm = graycoprops(glcm, 'homogeneity')[0]
    contrast = graycoprops(glcm, 'contrast')[0]
    correlation = graycoprops(glcm, 'correlation')[0]

    # gabung 16 fitur jadi 1 list
    features = np.concatenate([asm, idm, contrast, correlation])
    return features

