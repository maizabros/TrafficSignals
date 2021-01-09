import cv2
import numpy as np
import os
import sys

# Pequeña barra de progreso
class ProgressBar:
    def __init__(self, count, size=60, prefix="Computing"):
        self.count = count
        self.size = size
        self.prefix = prefix

    def show(self, j):  # Función para mostrar una barra de progreso
        x = int(self.size*j/self.count)
        sys.stdout.write("%s [%s%s%s] %i %%\r" 
            % (self.prefix, "#"*x, ">","."*(self.size-x), j*100/self.count))
        if j == self.count:
            sys.stdout.write("\n\n")
        sys.stdout.flush()

def create_dataset(folder, IMG_HEIGHT=83, IMG_WIDTH=83, COLOR=False):
    # Numero total de imagenes
    numImgs = sum([len(files) for r, d, files in os.walk(folder)])
    # Numero de clases
    numDirs = sum(os.path.isdir(os.path.join(folder, i)) for i in os.listdir(folder))
    print("Número de clases: ", numDirs)
    print("Número total de imágenes: ", numImgs)

    image_data = []
    class_name = []
    
    progrBar = ProgressBar(numImgs, prefix="Generating")
    aux = 0
    for directory in os.listdir(folder):
        for file in os.listdir(os.path.join(folder, directory)):
            imagePath = os.path.join(folder, directory,  file)
            if COLOR:
                image = cv2.imread(imagePath)
            else:
                image = cv2.imread(imagePath, 0)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
            image_data.append(image)
            class_name.append(int(directory))            
            aux += 1
            progrBar.show(aux)
    return np.uint8(image_data), class_name