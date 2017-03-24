# created by: Tobias Brockmeyer - tobias.brockmeyer@tum.de
# this code
# 1) reads an image of (usually 8) vials with chemical substances
# 2) optionally reduces the colors to, e.g., 32 colours
# 3) cuts segments out of the pictures: for every vial, one segment, size e.g.: h=805 px, w=100 px
# For each segment, the following operations are done:
# 4) resize to height of h (usually 100) px
# 5) depending on how many colors are detected, the threshold (~coarse or fine) is calculated with which the colored areas are distinguished from each other: if a vial is almost colorless, detection will have to be finer
# 6) areas of transition between colored areas are identified. In between, a line (position) is previsioned
# 7) if two lines have less than d pixels (e.g., 0.15*segment height) in between, a line between these two will replace them
# 8) if the are above and below one line are similar in colour, the line will be deleted

import numpy as np
import cv2
import itertools
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import array, uint8 , plot

#### packages for color reduction
from sklearn.cluster import MiniBatchKMeans
import argparse

img_matsum_y = [0]*8

### name of the picture, ideally in same path as this code file (optionally with path)
imagename = 'Image014.jpg'

img_0 = cv2.imread(imagename)

######## -------------
######## color reduction
######## this takes very long (~3 minutes).
######## Do it once, save picture, use this for further prototyping
######## and comment out the --colour reduction-- part
######
####ap = argparse.ArgumentParser()
####args = ['image', 256]
####args[0] = imagename
####
##### number of colours
####args[1] = 128
####
####print ("line 42")
####
##### load the image and grab its width and height
####image = img_0
####
####(h, w) = image.shape[:2]
####image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
##### reshape the image into a feature vector so that k-means
##### can be applied
####image = image.reshape((image.shape[0] * image.shape[1], 3))
####
##### apply k-means using the specified number of clusters and
##### then create the quantized image based on the predictions
########clt = MiniBatchKMeans(n_clusters = args["clusters"])
####clt = MiniBatchKMeans(n_clusters = args[1])
####labels = clt.fit_predict(image)
####quant = clt.cluster_centers_.astype("uint8")[labels]
####
##### reshape the feature vectors to images
####quant = quant.reshape((h, w, 3))
####image = image.reshape((h, w, 3))
#### 
##### convert from L*a*b* to RGB
####quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
####image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
#### 
####### display the images and wait for a keypress
######cv2.imshow("image", np.hstack([image, quant]))
######cv2.imwrite("result010.png", quant)
######cv2.waitKey(0)
####
####img_0 = quant
####
######## end color reduction
######## -------------

img_lines = img_0
# img = img_0[240:740, 480:660]
edges_up_left_X = [865]*8
edges_up_left_Y = [130,545,930,1330,1720,2110,2500,2890]
#edges_up_left_Y = [163,339,512,682,850,1020,1200,1369]
edges_low_right_X = [1690]*8
edges_low_right_Y = [230,645,1030,1430,1820,2210,2600,2990]
#edges_low_right_Y = [339,512,682,850,1020,1200,1369,1550]

print ("line 87")

img = [0]*8
for i in range (0, 8):
    #print(edges_up_left_X[i],":")
    #print(edges_low_right_X[i],"    ")
    #print(edges_up_left_Y[i], ":")
    #print(edges_low_right_Y[i],"    ")
    img[i] = img_0[edges_up_left_X[i]:edges_low_right_X[i], edges_up_left_Y[i]:edges_low_right_Y[i]]

for m in range (0, 8):
### image resize code 
    ### from: http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/ 
     
    # which height shall the new picture have? (in pixels) 
    h = 100 
     
    r = int(h) / img[m].shape[0] 
    dim = (int(img[m].shape[1] * r), int(h)) 
    ##print(img[m].shape) 
    ##print(r) 
    ##print(dim) 
     
    ### show resized picture 
     
    img[m] = cv2.resize(img[m], dim, interpolation = cv2.INTER_AREA) 
    ##cv2.imshow("resized", img[m]) 
    ##cv2.waitKey(0) 
     
    ### end show resized picture 
     
    ### plot colours 
     
    ###### prepare matrices for plots from image-matrix "img[m]": 
    ###### first, rows of colour information per pixel are summed up 
     
    img_matsum_x = np.sum(img[m], axis=1)
    img_matsum_y[m] = np.sum(img_matsum_x, axis=0)
    img_matsum_y_RGB = np.sum(img_matsum_y[m], axis = 0)
    #### definition of threshold (used further below, see comment "threshold")
    th = 0.7-(1/1000000)*img_matsum_y_RGB
##    if img_matsum_y_RGB > 570000:
##        th = 0.1
##    else:
##        th = 0.2
    print ("line 132") 
    ###### now, red, green and blue information is separated 
    img_matsum_r = list(itertools.chain.from_iterable(img_matsum_x[:,[0]])) 
    img_matsum_g = list(itertools.chain.from_iterable(img_matsum_x[:,[1]])) 
    img_matsum_b = list(itertools.chain.from_iterable(img_matsum_x[:,[2]])) 
     
    ###### create a [1,2,3,...,h]-matrix for ground axis of plot 
    ascending_numbers = [1] 
    for i in range(0,h-1): 
        ascending_numbers.append(ascending_numbers[i]+1) 
     
    ###### calculate moving averages 
        #### and set "points of extreme events" 
        #### (if value differs from current mean value considerably) 
        #### threshold (relevant deviation of value from current mean) 
        #### --defined further above, see comment "definition of threshold"
        
        #### amplification factor for separation lines in plot (black line): 
    amp_fac = 2500 # currently same factor for all colours; could be changed! 
    emptylist = [0]*h 
    ######### red colour: 
    mov_avg_r = [img_matsum_r[0]] 
    global u 
    u = 0 
    #for i in range(1,4): 
    for i in range(1,len(img_matsum_r)): 
        # list of colour values with current length i 
        img_matsum_r_tmp = [img_matsum_r[u:i]] 
        # calculate current (moving) average OF COLOUR VALUES 
        mov_avg_tmp = np.mean(img_matsum_r_tmp) 
        # calculate DEVIATION from current moving average 
        if mov_avg_tmp != 0:
            mov_avg_formula = (img_matsum_r[i]-mov_avg_tmp)/mov_avg_tmp
        else:
            mov_avg_formula = 0
        #print ("mov_avg_formula: ", mov_avg_formula) 
        mov_avg_r.append(mov_avg_tmp) 
        if (abs(mov_avg_formula))>th: 
            emptylist[i] += abs(mov_avg_formula)*amp_fac 
    ##        global u 
            u = i 
            # print ("u: ", u) 
     
    u = 0         
    ######### green colour: 
    mov_avg_g = [img_matsum_g[0]] 
    for i in range(1,len(img_matsum_g)): 
        # list of colour values with current length i 
        img_matsum_g_tmp = [img_matsum_g[u:i]] 
        mov_avg_tmp = np.mean(img_matsum_g_tmp) 
        mov_avg_formula = (img_matsum_g[i]-mov_avg_tmp)/mov_avg_tmp 
        mov_avg_g.append(mov_avg_tmp) 
        if (abs(mov_avg_formula))>th: 
            emptylist[i] += abs(mov_avg_formula)*amp_fac 
    ##        global u 
            u = i 
    print ("line 188") 
    u = 0 
    ######### blue colour: 
    mov_avg_b = [img_matsum_b[0]] 
    for i in range(1,len(img_matsum_b)): 
        # list of colour values with current length i 
        img_matsum_b_tmp = [img_matsum_b[u:i]] 
        mov_avg_tmp = np.mean(img_matsum_b_tmp) 
        mov_avg_formula = (img_matsum_b[i]-mov_avg_tmp)/mov_avg_tmp 
        mov_avg_b.append(mov_avg_tmp) 
        if (abs(mov_avg_formula))>th: 
            emptylist[i] += abs(mov_avg_formula)*amp_fac 
    ##        global u 
            u = i 
     
    ###### end calculate moving averages 
     
    ### smooth emptylist values: substitute "0"-values between two high values 
     
    emptylist_smoothed = [0]*len(emptylist) 
    # threshold e_mean: 
    th_em = 0.25 
    e_mean = th_em*np.mean(emptylist) 
     
    for i in range (0, img[m].shape[0]): 
        if i < img[m].shape[0]-1: 
            if emptylist[i-1] > th_em*e_mean and emptylist[i]==0 and emptylist[i+1] > th_em*e_mean: 
                emptylist_smoothed[i] = (emptylist[i-1]+emptylist[i+1])/2 
            else: 
                emptylist_smoothed[i] = emptylist[i] 
     
    ### end smooth emptylist values 
    print ("line 220") 
    plt.plot(ascending_numbers, img_matsum_r, 'r-') 
    plt.plot(ascending_numbers, img_matsum_g, 'g-')
    plt.plot(ascending_numbers, img_matsum_b, 'b-') 
     
    plt.plot(ascending_numbers, mov_avg_r, 'm-') 
    plt.plot(ascending_numbers, mov_avg_g, 'y-') 
    plt.plot(ascending_numbers, mov_avg_b, 'c-') 
     
    plt.plot(ascending_numbers, emptylist_smoothed, 'k-') 
     
    plt.axis([0, h+1, 0, 10000])
    #savefig('plot_%s.png' % (str(m)))
    #fig = plt.figure()
    #fig.savefig('plot_%s.png' % (str(m)))
    #plt.show() 
     
    ### end plot colours 
     
    # end image resize code 
     
    ### determine local maxima 
    ### in any "area" encirled by "0"s, determine maximum and save 
    ### position in an array 
     
    ### determine local maxima 
    line_position = [] 
    global pos_tmp 
    pos_tmp = 0 
    for i in range (0, img[m].shape[0]): 
        print ("line 250")
        #print ("i: ", i, "   value: ", emptylist_smoothed[pos_tmp]) 
        if emptylist_smoothed[i] != 0: 
            if emptylist_smoothed[i] > emptylist_smoothed[pos_tmp]: 
                pos_tmp = i 
                #print ("It's bigger:", emptylist[i]) 
            if i < img[m].shape[0] and emptylist_smoothed[i+1] == 0: 
                    line_position.append(pos_tmp) 
                    #line_position.append([emptylist_smoothed[pos_tmp]]) 
                    pos_tmp = 0 
            elif i == img[m].shape[0]: 
                    line_position.append(pos_tmp) 
                    #line_position.append([emptylist_smoothed[pos_tmp]]) 
                    pos_tmp = 0 
     
#########################for n in range (0, 8):
    print("before line_position = ",line_position)

#### modify line positions acc. to initial descriptions 7) and 8):

    # in which range radius (in pixels) should lines be joined?

    max_dist = 10
    
    line_position_new = [0]*(max(line_position)+1)
    line_position_new2 = [0]*(max(line_position)+1)
    for x in range (0,len(line_position)):
        #print (x)
        pos_tmp = line_position[x-1]
        line_position_new[pos_tmp] = 1

    add_zeros = [0]*max_dist
    line_position_new += add_zeros
    line_position_new2 += add_zeros

    # write all 1s detected in range into a separate cutout_tmp
    # calculate where new line shall be: [1,0,0,0,1,0,0] --> [0,0,1,0,0,0,0] (on pos 2)
    # [1,1,0,0,1,0,0] --> [0,0,1,0,0,0,0] (on pos 2)
    # [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1] --> [0,0,0,0,1,0...0]
    print ("line 289")

    for a in range (0,len(line_position_new)):
        #print(a)
        #print(line_position_new)
        if line_position_new[a] == 1:
            # cutout_tmp = []*(2*max_dist+1)
            newpos_list = [0,0]
            if len(line_position_new)-a-1<max_dist:
                end_b = len(line_position_new)-a-1
            else:
                end_b = max_dist+1
            for b in range (-max_dist, end_b):
                print ("current b = ",b)
                print ("current m = ",m)
                if line_position_new[a+b]>0:
                    newpos_list[0]+=b+max_dist+1
                    newpos_list[1]+=1
                    print(newpos_list)
                #line_position_new[a+b] = 0
            newpos = int((newpos_list[0]/newpos_list[1])-1)
            line_position_new2[newpos+a-max_dist]=1
            #print (line_position_new2)

    line_position_transfer = []
    for e in range(0,len(line_position_new2)):
        if line_position_new2[e] == 1:
            line_position_transfer.append(e)
    
    line_position = line_position_transfer

    print ("afterwards line_position = ",line_position)

#### end modify line positions

### draw lines in image where phase separation takes place  
 
    ###### prepare line parameters 
    number_of_lines = len(line_position) 
    left_image_border = img[m].shape[1] 
 
    ######### origin coordinates: 
    global orY,orX,endY,endX 
    orY = [0]*number_of_lines 
    orX = [0]*number_of_lines 
    for i in range (0, number_of_lines): 
        orX[i] = int(line_position[i]/h*(edges_low_right_X[m]-edges_up_left_X[m]))+int(edges_up_left_X[m])
        orY[i] = edges_up_left_Y[m]
    ######### end coordinates: 
    endY = [left_image_border]*number_of_lines 
    endX = [0]*number_of_lines 
    for i in range (0, number_of_lines): 
        endX[i] = orX[i]
        endY[i] = edges_low_right_Y[m]
     
    ###### draw lines 
    for i in range (0, number_of_lines): 
        img_lines = cv2.line(img_lines,(orY[i],orX[i]),(endY[i],endX[i]),(255,0,0),2) 
    # img_lines = cv2.line(img_lines,(orY[1],orX[1]),(endY[1],endX[1]),(255,0,0),5) 
    print ("line 343")
#### show image
## cv2.imshow("lines", img_lines) 
## cv2.waitKey(0) 

#### save image
    # imgfilename = "C:\\Users\\Tobias\\Art\\2-Phase\\2017-03-07 ImageRecog tryout\\Python Tobias\\Sensitivity020"
    # imgfilename += "\\result.png"
imgfilename = "result.png"
cv2.imwrite(imgfilename,img_lines)
