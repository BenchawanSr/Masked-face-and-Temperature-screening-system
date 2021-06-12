import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions

class2text = {
    0: 'Masked',
    1: 'No Masked',
}
num_classes = len(class2text)
model = tf.keras.models.load_model('test5.h5')
th = 0.85;

import cv2
import numpy as np

obj_cascade = cv2.CascadeClassifier('objtest1.xml')

# Load pre-trained model:
# net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

def demask(img):
    status = ""
    (h, w) = img.shape[:2]

    # Create 4-dimensional blob from image:
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104., 117., 123.], False, False)

    # Set the blob as input and obtain the detections:
    net.setInput(blob)
    detections = net.forward()

    # Initialize the number of detected faces counter detected_faces:
    detected_faces = 0

    af=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    pf=[(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)]
    
    # Iterate over all detections: 
    for i in range(0, detections.shape[2]): #วิ่งตามกล้องที่ตรวจจับหน้าได้ทุกหน้า
        # Get the confidence (probability) of the current detection:
        confidence = detections[0, 0, i, 2]

        # Only consider detections if confidence is greater than a fixed minimum confidence:
        if confidence > 0.7:
            # Increment the number of detected faces:
            detected_faces += 1 #จำนวนคนที่ยืนต่อหน้ากล้อง
            
            # Get the coordinates of the current detection:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            hh=endY-startY
            ww=endX-startX
            areaf=ww*hh
            af[detected_faces-1]=areaf
            pf[detected_faces-1]=(startX, startY, endX, endY)

            # Draw the detection and the confidence:
            text = "{:.3f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            #cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
            #cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            #hh=endY-startY
            #image2=image[startY+int(hh/2): endY,startX:endX]
            #cv2.imshow("image2",image2)
    
    #print(detected_faces)
    #print(af)
    idxmf=np.argmax(af)
    fp=pf[idxmf]
    #print(fp)
    sxmax=fp[0]
    symax=fp[1]
    exmax=fp[2]
    eymax=fp[3]
    hhhh=eymax-symax
    if((exmax-sxmax)<640)and((eymax-symax)<480)and(sxmax>0)and(symax>0):
        
        #print(sxmax)
        #print(symax)
        #print(exmax)
        #print(eymax)
        #cv2.imwrite('testobj.jpg',img[symax+int(0.5*(eymax-symax)):eymax,sxmax:exmax])
        gray = cv2.cvtColor(img[symax+int(0.5*(eymax-symax)):eymax,sxmax:exmax], cv2.COLOR_BGR2GRAY)
    
        obj = obj_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(30,30))
        #print(len(obj))
        if (len(obj) > 0):
            for (xm,ym,wm,hm) in obj:
                #cv2.rectangle(img,(xm+sxmax,ym+symax+int(0.5*(eymax-symax))),(xm+sxmax+wm,ym+symax+int(0.5*(eymax-symax))+hm),(0,0,255),5)
                cv2.putText(img,"No masked",(sxmax,symax-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                status = "NM"
                cv2.rectangle(img, (sxmax, symax), (exmax, eymax), (0, 0, 255),3)
        else:
            cv2.imwrite('testobj.jpg',img[symax:eymax,sxmax:exmax])
            files = ['testobj.jpg']
            images = []
            ##print(type(images))
            for f in files:
                # Load and resize the image
                img224 = image.load_img(f, target_size=(224, 224))
                # Convert the Image object into a numpy array
                img224 = image.img_to_array(img224)
                # Add to a list of images
                images.append(img224)
            images = np.asarray(images)            
            ##print(images.shape)
            #print(images)         
            # Preprocess the input array
            xx = preprocess_input(images)
            #print(f"Image shape: {x.shape}")
            # Feed the preprocessed, downloaded image to the pretrained VGG-16.
            # The outputs are the probabilities of classes defined in ImageNet.
            probs = model.predict(xx)
            probsone = probs[0]
            indexmax = np.argmax(probsone)
            valmax = np.max(probsone)
            #print(indexmax)
            if (valmax > th):
                name = class2text[indexmax]
                if (class2text[indexmax]=='Masked'):
                    cv2.putText(img,name,(sxmax,symax-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                    status = "M"
                    cv2.rectangle(img, (sxmax, symax), (exmax, eymax), (0, 255, 0),3)

                elif (class2text[indexmax]=='No Masked'):
                    cv2.putText(img,name,(sxmax,symax-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                    #status = "NM"
                    cv2.rectangle(img, (sxmax, symax), (exmax, eymax), (0, 0, 255),3)
            else:
                name = "Unknow"
                cv2.putText(img,name,(sxmax,symax-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                cv2.rectangle(img, (sxmax, symax), (exmax, eymax), (255, 255, 255),3)

                files = ['testobj.jpg']
            images = []
            ##print(type(images))
            for f in files:
                # Load and resize the image
                img224 = image.load_img(f, target_size=(224, 224))
                # Convert the Image object into a numpy array
                img224 = image.img_to_array(img224)
                # Add to a list of images
                images.append(img224)
            images = np.asarray(images)            
            ##print(images.shape)
            #print(images)         
            # Preprocess the input array
            xx = preprocess_input(images)
            #print(f"Image shape: {x.shape}")
            # Feed the preprocessed, downloaded image to the pretrained VGG-16.
            # The outputs are the probabilities of classes defined in ImageNet.
            probs = model.predict(xx)
            probsone = probs[0]
            indexmax = np.argmax(probsone)
            valmax = np.max(probsone)
            #print(indexmax)
            '''
            if (valmax > th):
                name = class2text[indexmax] + " " + str(int(valmax*100)) + "%"
                if (class2text[indexmax]=='Masked'):
                    cv2.putText(img,name,(sxmax,symax-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                    status = "M"
                    cv2.rectangle(img, (sxmax, symax), (exmax, eymax), (0, 255, 0),3)
                elif (class2text[indexmax]=='No Masked'):
                    cv2.putText(img,name,(sxmax,symax-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                    #status = "NM"
                    cv2.rectangle(img, (sxmax, symax), (exmax, eymax), (0, 0, 255),3)
            else:
                name = "Unknow"
                cv2.putText(img,name,(sxmax,symax-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                cv2.rectangle(img, (sxmax, symax), (exmax, eymax), (255, 255, 255),3)
            '''
        
    #print(pf)
    #cv2.imshow("image",img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
    return img,status
