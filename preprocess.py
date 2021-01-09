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

def gradient(img):
    Ix = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)    
    E = np.sqrt(Ix ** 2 + Iy ** 2)
    Phi = np.rad2deg(np.arctan2(Iy, Ix))
    return E, Phi

def nonMaxSuppress(M, alpha):
    imEt = np.zeros(alpha.shape)
    EN = np.zeros(M.shape)

    imEt[((alpha>=-22.5) & (alpha<22.5)) | ((alpha>=-180) & (alpha<-157.5)) | ((alpha>=157.5) & (alpha<180))] = 1
    imEt[((alpha>=22.5) & (alpha<67.5)) | ((alpha>=-157.5) & (alpha<-112.5))] = 2
    imEt[((alpha>=67.5) & (alpha<112.5)) | ((alpha>=-112.5) & (alpha<-67.5))] = 3
    imEt[((alpha>=112.5) & (alpha<157.5)) | ((alpha>=-67.5) & (alpha<-22.5))] = 4
    
    ofst = 1
    M_amp = cv2.copyMakeBorder(M, ofst, ofst, ofst, ofst, cv2.BORDER_REPLICATE)
    
    et1 = np.array(np.where(imEt == 1)).T + ofst
    et2 = np.array(np.where(imEt == 2)).T + ofst
    et3 = np.array(np.where(imEt == 3)).T + ofst
    et4 = np.array(np.where(imEt == 4)).T + ofst
        
    vec1 = np.vstack((M_amp[et1[:,0]-1, et1[:,1]  ], M_amp[et1[:,0]+1, et1[:,1]  ])).T
    vec2 = np.vstack((M_amp[et2[:,0]-1, et2[:,1]-1], M_amp[et2[:,0]+1, et2[:,1]+1])).T
    vec3 = np.vstack((M_amp[et3[:,0]  , et3[:,1]-1], M_amp[et3[:,0]  , et3[:,1]+1])).T
    vec4 = np.vstack((M_amp[et4[:,0]+1, et4[:,1]-1], M_amp[et4[:,0]-1, et4[:,1]+1])).T
    
    zeros1 = ((M_amp[et1[:,0], et1[:,1]] > vec1[:,0]) & (M_amp[et1[:,0], et1[:,1]] > vec1[:,1]))
    zeros2 = ((M_amp[et2[:,0], et2[:,1]] > vec2[:,0]) & (M_amp[et2[:,0], et2[:,1]] > vec2[:,1]))
    zeros3 = ((M_amp[et3[:,0], et3[:,1]] > vec3[:,0]) & (M_amp[et3[:,0], et3[:,1]] > vec3[:,1]))
    zeros4 = ((M_amp[et4[:,0], et4[:,1]] > vec4[:,0]) & (M_amp[et4[:,0], et4[:,1]] > vec4[:,1]))

    EN[et1[zeros1, 0] - ofst, et1[zeros1, 1] - ofst] = M[et1[zeros1, 0] - ofst, et1[zeros1, 1] - ofst]
    EN[et2[zeros2, 0] - ofst, et2[zeros2, 1] - ofst] = M[et2[zeros2, 0] - ofst, et2[zeros2, 1] - ofst]
    EN[et3[zeros3, 0] - ofst, et3[zeros3, 1] - ofst] = M[et3[zeros3, 0] - ofst, et3[zeros3, 1] - ofst]
    EN[et4[zeros4, 0] - ofst, et4[zeros4, 1] - ofst] = M[et4[zeros4, 0] - ofst, et4[zeros4, 1] - ofst]
   
    return EN

def imageDistortion(img_class):
    distImgs = np.zeros_like(img_class)
    dx, dy = img_class.shape[1:3]
    dxr, dyr = dx//2, dy//2
    for i in range(len(img_class)):
        img = img_class[i]
        
        # Rotación de la imagen
        angle = (15*2*(np.random.rand()-.5)) # angulo de -15 a 15 grados
        rot = cv2.getRotationMatrix2D((dxr,dyr),angle,1)
        rot[0,2] = 2*2*(np.random.rand()-.5) # trasladar x desde -2 hasta 2 pixeles
        rot[1,2] = 2*2*(np.random.rand()-.5) # trasladar y desde -2 hasta 2 pixeles
        warpImg = cv2.warpAffine(img,rot,(dx,dy), flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Transformiación de perspectiva
        d0 = 2*(np.random.rand() - 0.5)
        d1 = 2*(np.random.rand() - 0.5)
        p1 = np.float32([[2,2],[30,2],[0,30],[30,30]])
        p2 = np.float32([[d0,d1],[30+d0,d1],[d0,30+d1],[30+d0,30+d1]])
        persp = cv2.getPerspectiveTransform(p1,p2)
        
        distImgs[i] = cv2.warpPerspective(warpImg,persp,(dx,dy), flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return distImgs

def ampliacion(imagen):
    al, ah = imagen.min(), imagen.max()
    imagen = np.float32(imagen)
    imagen = 255*(imagen-al)/(ah-al)
    return np.uint8(imagen)

def processImages(img_data, gauss=True, sobel=True, eq=True, nonMaxS=False, amp=True):
    if (img_data.shape[-1] == 3) & (len(img_data.shape) < 4):
        img_data = img_data.reshape((1,img_data.shape[0], img_data.shape[1], img_data.shape[2]))
    elif len(img_data.shape) < 3:
        img_data = img_data.reshape((1,img_data.shape[0], img_data.shape[1]))
    if img_data.shape[0] > 1:
        progrBar = ProgressBar(img_data.shape[0]//2, prefix="Processing")
    for i in range(img_data.shape[0]):
        if amp:
            img_data[i] = ampliacion(img_data[i])
        if eq:
            if len(img_data[i].shape) == 3:
                img_to_yuv = cv2.cvtColor(img_data[i],cv2.COLOR_BGR2YUV)
                img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
                img_data[i] = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
            else:
                img_data[i] = cv2.equalizeHist(img_data[i])
        if gauss:
            img_data[i] = cv2.GaussianBlur(img_data[i], (3,3), 1)
        if sobel:
            img_data[i], Phi = gradient(img_data[i])
        if nonMaxS:
            img_data[i] = nonMaxSuppress(img_data[i],  Phi)
        if img_data.shape[0] > 1:
            if i % 2 == 0:
                progrBar.show(i//2+1)
    return img_data
