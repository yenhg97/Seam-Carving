import cv2
import numpy as np

class SeamCarving:
        def __init__(self, fileName, outWidth, outHeight):
                self.fileName = fileName
                self.outWidth = outWidth
                self.outHeight = outHeight
                
                self.inImg = cv2.imread(fileName).astype(np.float64)
                self.inHeight, self.inWidth = self.inImg.shape[:2]
                
                self.outImg = np.copy(self.inImg)
                
                self.Start()
                
        
        def Start(self):
                deltaRow = int(self.outHeight - self.inHeight)
                deltaCol = int(self.outWidth - self.inWidth)
        
                if deltaCol > 0:
                        self.Enlarge(deltaCol)
                elif deltaCol < 0:
                        self.Reduce(deltaCol * -1)
                
                if deltaRow > 0:
                        self.outImg = self.Rotate(self.outImg, 1)
                        self.Enlarge(deltaRow)
                        self.outImg = self.Rotate(self.outImg, 0)
                elif deltaRow < 0:
                        self.outImg = self.Rotate(self.outImg, 1)
                        self.Reduce(deltaRow * -1)
                        self.outImg = self.Rotate(self.outImg, 0)
    
        def Reduce(self, n):
                for i in range(n):
                        energyMap =  self.Energy()
                        
                        # Dynamic Programming
                        m, n = energyMap.shape                        
                        M = np.copy(energyMap)
                        for row in range(1, m):
                            for col in range(n):
                                M[row, col] = energyMap[row, col] + np.amin(M[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
                        
                        # Backtracking
                        seam = np.zeros((m,), dtype=np.uint32)
                        seam[-1] = np.argmin(M[-1])
                        for row in range(m - 2, -1, -1):
                            pre_x = seam[row + 1]
                            if pre_x == 0:
                                seam[row] = np.argmin(M[row, : 2])
                            else:
                                seam[row] = np.argmin(M[row, pre_x - 1: min(pre_x + 2, n - 1)]) + pre_x - 1
                        
                        self.deleteSeam(seam)
        
        def Enlarge(self, n):
                temp = np.copy(self.outImg)
                seamsRecord = []
                
                for i in range(n):
                        energyMap = self.Energy()
                        
                        # Dynamic Programming
                        m, n = energyMap.shape
                        M = np.copy(energyMap)
                        for row in range(1, m):
                            for col in range(n):
                                M[row, col] = energyMap[row, col] + np.amin(M[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
                       
                        # Backtracking
                        seam = np.zeros((m,), dtype=np.uint32)
                        seam[-1] = np.argmin(M[-1])
                        for row in range(m - 2, -1, -1):
                            pre_x = seam[row + 1]
                            if pre_x == 0:
                                seam[row] = np.argmin(M[row, : 2])
                            else:
                                seam[row] = np.argmin(M[row, pre_x - 1: min(pre_x + 2, n - 1)]) + pre_x - 1
                        
                        seamsRecord.append(seam)
                        self.deleteSeam(seam)

                self.outImg = np.copy(temp)
                for i in range(len(seamsRecord)):
                        seam = seamsRecord.pop(0)
                        self.addSeam(seam)

        
        def Energy(self):
                blue, green, red = cv2.split(self.outImg)
                blueEnergy = np.absolute(cv2.Scharr(blue, -1, 1, 0)) + np.absolute(cv2.Scharr(blue, -1, 0, 1))
                greenEnergy = np.absolute(cv2.Scharr(green, -1, 1, 0)) + np.absolute(cv2.Scharr(green, -1, 0, 1))
                redEnergy = np.absolute(cv2.Scharr(red, -1, 1, 0)) + np.absolute(cv2.Scharr(red, -1, 0, 1))
                return blueEnergy + greenEnergy + redEnergy
                

        def deleteSeam(self, seam):
                m, n = self.outImg.shape[:2]
                output = np.zeros((m, n - 1, 3))
                for row in range(m):
                    col = seam[row]
                    for ch in range(3):
                        output[row, :, ch] = np.delete(self.outImg[row, :, ch], [col])
                self.outImg = np.copy(output)


        def addSeam(self, seam):
                m, n = self.outImg.shape[:2]
                output = np.zeros((m, n + 1, 3))
                for row in range(m):
                    col = seam[row]
                    for ch in range(3):
                        if col == 0:
                            p = np.average(self.outImg[row, col: col + 2, ch])
                            output[row, col, ch] = self.outImg[row, col, ch]
                            output[row, col + 1, ch] = p
                            output[row, col + 1:, ch] = self.outImg[row, col:, ch]
                        else:
                            p = np.average(self.outImg[row, col - 1: col + 1, ch])
                            output[row, : col, ch] = self.outImg[row, : col, ch]
                            output[row, col, ch] = p
                            output[row, col + 1:, ch] = self.outImg[row, col:, ch]
                self.outImg = np.copy(output)


        def Rotate(self, img, index):
                m, n, channel = img.shape
                output = np.zeros((n, m, channel))
                if index:
                    imgFlip = np.fliplr(img)
                    for ch in range(channel):
                        for row in range(m):
                            output[:, row, ch] = imgFlip[row, :, ch]
                else:   
                    for ch in range(channel):
                        for row in range(m):
                            output[:, m - 1 - row, ch] = img[row, :, ch]
                return output

        
        def Save(self, fileName):
                cv2.imwrite(fileName, self.outImg.astype(np.uint8))
        
        def Show(self):
                cv2.imshow('image_before', self.inImg.astype(np.uint8))
                cv2.imshow('image_after', self.outImg.astype(np.uint8))
                cv2.waitKey()
                cv2.destroyAllWindows()
                
name = input("Enter image's name: ")
width = int(input("Enter the new width: "))
height = int(input("Enter the new height: "))

img = SeamCarving(name, width, height)
#img.Save('new.jpg')
img.Show()