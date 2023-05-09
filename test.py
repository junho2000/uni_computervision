import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_img(img, ang):
    # compute rotation matrix
    m60 = np.float32([[np.cos(ang), -1*np.sin(ang)],
                        [np.sin(ang), np.cos(ang)]])
    # compute output coordinates for image corners
    row, col = img.shape[:2] # 1124 843
    coordinate = np.array([[0,0], [0,row-1], [col-1, 0], [col-1, row-1]]) #(x,y)
    newcoordinate  = []
    for i in range(4):
        newcoordinate.append((m60@coordinate[i].T).T)
    newcoordinate = np.array(newcoordinate)
    print(newcoordinate)
    col_min, row_min = np.min(newcoordinate, axis=0).astype(int)
    col_max, row_max = np.max(newcoordinate, axis=0).astype(int)
    
    # compute output image size and offset
    col_out, row_out = col_max - col_min + 1, row_max - row_min + 1
    print('output(row, col) =', row_out, col_out)
    col_offset, row_offset = -col_min, -row_min
    print('offsets(x,y) =', col_offset, row_offset)
    
    # define output image object
    img_out = np.zeros((row_out, col_out, 3), dtype=img.dtype)
    
    # for each output image coordinate, 
    # compute corresponding original image coordinate
    # get pixel value from original image with interpolation
    for r in range(row_out):
        for c in range(col_out):
            c_org, r_org = np.array([c - col_offset, r - row_offset])@m60
            if c_org >= 0 and r_org >= 0 and c_org <= col - 1 and r_org <= row - 1:
                c_org_down, r_org_down = int(np.floor(c_org)), int(np.floor(r_org))
                c_org_up, r_org_up = c_org_down+1, r_org_down+1
                alpha, beta = c_org - c_org_down, r_org - r_org_down
                img_out[r, c] = (1-beta)*(1-alpha)*img[r_org_down,c_org_down] + (1-beta)*alpha*img[r_org_down,c_org_up] + beta*(1-alpha)*img[r_org_up,c_org_down] + beta*alpha*img[r_org_up,c_org_up]
                
    return img_out

# Load the image
img = cv2.imread("/Users/kimjunho/Desktop/컴퓨터비전3-1/[CV]A1/img_example.JPG")
# Set the rotation angle
ang = np.pi/3
img_out = rotate_img(img, ang)
cv2.imshow('img', img)
# Convert the image from BGR to RGB format
img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
# Display the image using Matplotlib
plt.imshow(img_out)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()