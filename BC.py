import ECS_ST
import cv2
from pyzbar import pyzbar
import argparse

def readBarcode():

    cap = cv2.VideoCapture(1)

    bb=0

    while(cap.isOpened()):
        
        ret, img = cap.read()
        '''
        channels = cv2.split(img)
        B_HE = cv2.equalizeHist(channels[0]) #B
        G_HE = cv2.equalizeHist(channels[1]) #G
        R_HE = cv2.equalizeHist(channels[2]) #R
        img = cv2.merge((B_HE,G_HE,R_HE))
        #img = cv2.flip(img,1)
        '''

        barcodes = pyzbar.decode(img)
        #print(barcodes)
        
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite("1.jpg",img)
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            #print(barcodeData)
            if len(barcodeData) == 8:
                if barcodeData == "09601413":
                    name="Benchawan_S"
                    #print(name)
                if barcodeData == "09601433":
                    name="Passaraporn_D"
                    #print(name)
                if barcodeData == "09601344":
                    name="Jameekorn_S"
                    #print(name)
                if barcodeData == "09601406":
                    name="Natthamon_U"
                    #print(name)
                if barcodeData == "09601341":
                    name="Komkrit_H"
                    #print(name)
                #ECS_ST.tts("ยินดีต้อนรับ คุณ"+name,'th')
               
                #cv2.putText(img, barcodeData, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 255, 0), 2)
                bb=1
        
        cv2.imshow("test1",img)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (bb==1):
            cv2.imwrite("testsave.jpg",img)
            break

    cap.release()
    
    cv2.destroyAllWindows()
    return barcodeData,name



