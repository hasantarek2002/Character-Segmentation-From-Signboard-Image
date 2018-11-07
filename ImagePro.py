import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
from skimage import exposure
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, opening
from skimage.segmentation import clear_border
import matplotlib.patches as mpatches
from skimage.color import label2rgb
from skimage.morphology import disk
from skimage.filters.rank import mean
from skimage.filters.rank import median
import numpy as np
from skimage.transform import resize
import cv2


#def increase_y ()
def slide(image):

    #print("Img", image.shape)
    # read the image and define the stepSize and window size 
    # (width,height)
    #image = cv2.imread("cell.png") # your image path
    tmp = image # for drawing a rectangle
    stepSize = 99
    (w_width, w_height) = (99, 99) # window size
    for x in range(0, image.shape[1] - w_width , stepSize):
        for y in range(0, image.shape[0] - w_height, stepSize):
            window = image[x:x + w_width, y:y + w_height, :]
            
            # classify content of the window with your classifier and  
            # determine if the window includes an object (cell) or not
            # draw window on image
            cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2) # draw rectangle on image
            plt.imshow(tmp)
            # show all windows
    plt.show()




camera1 = io.imread("1.jpg", as_gray=True)
camera1 = resize(camera1, (300, 600))

camera1 = exposure.equalize_adapthist(camera1)
camera1 = exposure.rescale_intensity(camera1)
camera1 = exposure.adjust_sigmoid(camera1)

camera =  rgb2gray(camera1)
camera = mean(camera, disk(1))
camera = median(camera, disk(1))
from scipy.misc import imsave
#print(camera)
val = filters.threshold_otsu(camera)
camera = closing(camera > val, square(2))
camera = opening(camera, square(2))
camera = clear_border(camera)
#camera = label(camera)

from scipy import ndimage as ndi
cam = ndi.binary_fill_holes(camera)
#camera_opn = opening(cam, square(5))

label_objects, nb_labels = ndi.label(cam)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 40
mask_sizes[0] = 0
cam_clear = mask_sizes[label_objects]
plt.imshow(cam_clear , cmap='gray', interpolation='nearest')
#camera2 = label(camera)
camera2 = label(cam_clear)
#print("DSDDD", camera2.shape)


'''
edges = canny(camera2, 2, 1, 25)
lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                 line_gap=3)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(camera, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')
ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, camera2.shape[1]))
ax[2].set_ylim((camera2.shape[0], 0))
ax[2].set_title('Probabilistic Hough')
'''
camera = label2rgb(camera2, image=camera1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(camera , cmap='gray', interpolation='nearest')
#ax.imshow(camera , cmap='gray', interpolation='nearest')
count = 0
regions = regionprops(camera2)
for region in regions:
    # take regions with large enough areas
    if region.area >= 650:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        slice_hei = int((maxr - minr)* 0.1)
        #print(minr, minc, maxr, maxc)
        croped = camera1[minr-slice_hei:maxr+slice_hei, minc:maxc]
        binary_crop = cam_clear[minr-slice_hei:maxr+slice_hei, minc:maxc]
        #print("Here",camera.shape)
        wid = int(((maxc - minc)/((maxr - minr)*(1.05)))*100)
        #print(wid)
        #croped1 = resize(croped, (wid, 100), anti_aliasing=True)
        #print(minr, minc, maxr, maxc)
        croped1 = resize(croped, (100, wid))
        binary_croped1 = resize(binary_crop, (100, wid)).astype(int)
        #plt.imshow(croped1)
        #print(binary_croped1)
        histogram = binary_croped1.sum(axis=1)
        maxi = np.amax(histogram)
        for i in range(50):
            if maxi-40 < histogram[i] < maxi+40:
                binary_croped1[i,:] = 0 
        #plt.plot(binary_croped1)
        
        binary_croped_label = label(binary_croped1)
        char_regions = regionprops(binary_croped_label)
        #cv2.imshow('image',binary_croped_label)
        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        #cv2.destroyWindow()
        #print(croped)
        #print(char_regions)
        #plt.ion()
        for char_reg in char_regions:
            #print(char_reg.area)
            if char_reg.area >= 650:
                 import random
                 change = random.random()%255
                 #print(change)
                 cminr, cminc, cmaxr, cmaxc = char_reg.bbox
                 char_slice_hei = int((cmaxr - cminr)* 0.1)
                 #print(minr, minc, maxr, maxc)
                 #plt.plot(croped1)
                 binary_crop = binary_croped1[cminr-char_slice_hei:cmaxr+char_slice_hei, cminc:cmaxc]
                 char_croped_si= croped1[0:100, cminc:cmaxc]
                 #print("Here",camera.shape)
                 cwid = int(((cmaxc - cminc)/((cmaxr - cminr)*(1.05)))*100)
                 
                 char_croped1 = resize(char_croped_si, (100, cwid))
                 char_binary_croped1 = resize(binary_crop, (100, wid)).astype(int)
                 plt.imshow(char_croped1)
                 io.imsave('charimage'+str(change)+'.png', char_croped1)
        plt.show()
                 #rect = mpatches.Rectangle((cminc, cminr), cmaxc - cminc, (cmaxr - cminr)*(1.05),
                                  #fill=False, edgecolor=(change,0,0), linewidth=2)
                                  
        
        #print(histogram)
        
        #Show
        
        
        #his= np.histogram(croped1)
        #print(his)
        #slide(croped1)
        #io.all_warnings()
        #io.imsave('image'+str(count)+'.png', croped1)
        count+=1
        #plt.imshow(croped , cmap='gray', interpolation='nearest')
        rect = mpatches.Rectangle((minc, minr), maxc - minc, (maxr - minr)*(1.05),
                                fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
ax.set_axis_off()
plt.tight_layout()
plt.show()
