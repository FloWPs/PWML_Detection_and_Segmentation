import cv2
import cc3d
import numpy as np
import torch
from torchvision import transforms



#############################
#   Traitement des donnees  #
#############################



def vox2mm3(vol):
    """
    Convert number of voxels to volume in mm^3
    """
    voxel_volume = 0.168**3
    return vol * voxel_volume



def crop_center(img,cropx_i,cropy_i,cropz_i,cropx_e,cropy_e,cropz_e):
    x,y,z = img.shape[0],img.shape[1],img.shape[2]
    
    startx = x//2-(cropx_i)
    starty = y//2-(cropy_i)
    startz = z//2-(cropz_i)
    
    endx = x//2+(cropx_e)
    endy = y//2+(cropy_e)
    endz = z//2+(cropz_e)  
    return img[startx:endx,starty:endy,startz:endz]



def crop_center_full(img, crop_parameters):
    x,y,z = img.shape[0],img.shape[1],img.shape[2]
    
    cropx_i, cropy_i, cropz_i, cropx_e, cropy_e, cropz_e = crop_parameters
    
    startx = x//2-(cropx_i)
    starty = y//2-(cropy_i)
    startz = z//2-(cropz_i)
    
    endx = x//2+(cropx_e)
    endy = y//2+(cropy_e)
    endz = z//2+(cropz_e)

    # print(startx, starty, startz, endx, endy, endz)
    return img[startx:endx, starty:endy, startz:endz]


#############################################
#      Connectect components extraction     #
#############################################


def extract_vignette(image, centroid, dimensions=(64, 64)):
    """
    Extract a vignette of given dimensions from an image
    centered around the centroid coordinates (connected component)
    """
    # print(centroid[1], centroid[0])
    x1 = int(centroid[0])-(dimensions[0]//2)
    x2 = int(centroid[0])+(dimensions[0]//2)
    y1 = int(centroid[1])-(dimensions[1]//2)
    y2 = int(centroid[1])+(dimensions[1]//2)
    # print(x1, x2, y1, y2)

    # Dimension verification
    if (x2-x1) != dimensions[0]:
        d = dimensions[0] - (x2-x1)
        x2 += d
    if (y2-y1) != dimensions[1]:
        d = dimensions[1] - (y2-y1)
        y2 += d

    # If vignette out of bounds
    if x1 < 0:
        # print(1)
        x2 -= x1
        x1 -= x1
    if x2 > image.shape[0]:
        # print(2)
        x1 += (image.shape[0])-x2
        x2 += (image.shape[0])-x2
    if y1 < 0:
        # print(3)
        y2 -= y1
        y1 -= y1
        
    if y2 > image.shape[1]:
        # print(4)
        y1 += (image.shape[1])-y2
        y2 += (image.shape[1])-y2
    
    vignette = image[x1:x2, y1:y2]
    coordinates = [[x1, x2], [y1, y2]]
    # plt.imshow(vignette, 'gray')
    # plt.show()

    assert vignette.shape == dimensions

    return vignette #, coordinates


def read_and_preprocess_multiview(imgS, imgC):

    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imageS = imgS.astype(np.uint8)
    imageC = imgC.astype(np.uint8)

    imageS = cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB) # sagittal slice only
    imageC = cv2.cvtColor(imageC, cv2.COLOR_BGR2RGB) # coronal slice only
    
    img_torchS = val_transforms(imageS)
    img_torchC = val_transforms(imageC)

    imageSC = [img_torchS, img_torchC]
    imageSC = torch.cat(imageSC, dim=0)
    img_torchSC = torch.unsqueeze(imageSC, 0)

    return img_torchSC



##############################
#    Seuillage des lésions   #
##############################



def remove_small_lesions(mask, percentage=0.9):
    """
    Only keep 90% of the biggest lesions from a segmentation map (3D Mask)
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    lc = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesion found after crop.')
        return mask, None, None
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # Total number of lesion voxels
    lesion_total = np.sum(mask)
    # Initiate cumulated lesion volume
    volume = 0
    
    # print('LESION TOTAL :', lesion_total)
    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # If the biggest lesion is larger than 90% of the lesional volume
        if lesions[k] >= lesion_total*percentage and c == 0 and N > 1:
            # print(k, ': 1ST IF')
            smallest_lesion_size = lesions[k]
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc +=1
            c = 1
        # print(k, lesions[k])
        volume += lesions[k]
        # Assure that we keep 90% of the biggest lesions by the end
        if volume <= lesion_total*percentage and N > 1: # MODIF FLORA FROM < TO <=
            # print(k, '2ND IF')
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            # print('keep lesion', k, volume)
            
    # Condition to keep if only 1 lesion
    if N == 1:
        # print(k, '3RD IF')
        lesion_cluster = np.where(output == 1)
        new_mask[lesion_cluster] = 1
        lc += 1
    
    if N > 1:
        smallest_lesion_size = lesion_cluster[0].shape[0]
    else: # Only 1 lesion
        smallest_lesion_size = lesion_total

    # print(f'{lc} lesions kept over {N} in total (min lesion size = {smallest_lesion_size} voxels or {round(vox2mm3(smallest_lesion_size), 2)} mm3)')

    return new_mask, smallest_lesion_size, round(vox2mm3(smallest_lesion_size), 2)



def remove_small_lesions_threshold(mask, min_lesion_size=10):
    """
    Only keep lesions bigger than `min_lesion_size` from a segmentation map (3D Mask)
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    lc = 0
    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesion found after crop.')
        return mask #, None, None
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # Total number of lesion voxels
    lesion_total = np.sum(mask)
    # Initiate cumulated lesion volume
    volume = 0
    
    c = 0
    for k in sorted(lesions, key=lesions.get, reverse=True):
        # Assure that we keep lesions bigger than threshold
        # print(k, vox2mm3(lesions[k]))
        # if vox2mm3(lesions[k]) > min_lesion_size:
        if lesions[k] > min_lesion_size:
            lesion_cluster = np.where(output == k)
            new_mask[lesion_cluster] = 1
            lc += 1
            volume += lesions[k]
            # print('keep lesion', k, volume)
            c = 1
    
    # if at least one lesion was above threshold
    if N > 1 and c == 1:
        smallest_lesion_size = lesion_cluster[0].shape[0]
    
    # if there is no lesion above the threshold
    if N > 1 and c == 0:
        print('NEW MASK WITH NO LESIONS ABOVE THRESHOLD :', np.max(new_mask))
        return new_mask #, None, None

    if N == 1: # Only 1 lesion
        smallest_lesion_size = lesion_total

    # print(lc, 'lesions kept over', N, 'in total (min lesion size =', smallest_lesion_size, 'voxels or', round(vox2mm3(smallest_lesion_size), 2), 'mm3)')
    # print(f'{round(volume/lesion_total*100, 1)}% of the lesional volume remaining.')
    return new_mask #, smallest_lesion_size, round(vox2mm3(smallest_lesion_size), 2)


def filter_with_PC_intensities(volume, mask, threshold, pred=False):
    """
    Only keep lesions where the median intensity is > to the intensity threshold value of the plexus choroid.
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)

    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(mask, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesion found in this mask.')
        return mask #, None, None
        
    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]
    
    kept = 0
    vol = 0

    for k in sorted(lesions, key=lesions.get, reverse=True):
        # Get the pixels belonging to each lesion
        lesion_cluster = np.where(output == k)
        # print(k, np.min(volume[lesion_cluster]), np.mean(volume[lesion_cluster]), np.median(volume[lesion_cluster]), np.max(volume[lesion_cluster]))
        
        # If the median intensity of the current lesion is > to PC intensity threshold
        # if np.median(volume[lesion_cluster]) >= threshold:
        if (pred == True and np.median(volume[lesion_cluster]) >= 75 and np.median(volume[lesion_cluster]) <= 125)\
            or (pred == False and np.median(volume[lesion_cluster]) >= threshold):
            # print(pred)
            # We keep the lesion
            new_mask[lesion_cluster] = 1
            kept += 1
            vol += lesions[k]
        else:
            print(f'[INFO] Dumping lesion {k}. Size : {vox2mm3(lesions[k])} mm3')

    # print(kept, 'lesions kept over', N, 'in total.')
    # if N == 1:
    #     print(f'Size of unique PWML : {vox2mm3(lesions[k])} mm3')
    # print(f'{round(vol/np.sum(mask)*100, 1)}% of the lesional volume remaining.\n')

    return new_mask



#######################
#      Inférence      #
#######################



def thresh_func(mask, thresh=0.5):
    mask[mask >= thresh] = 1
    mask[mask < thresh] = 0

    return mask


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = torch.sum(pred * target)
    pred_sum = torch.sum(pred * pred)
    target_sum = torch.sum(target * target)

    return 1 - ((2. * intersection + 1e-5) / (pred_sum + target_sum + 1e-5))



############################
#      Visualisation       #
############################



def insert_mask(full_brain_pred, crop_pred, crop_parameters):
    """
    Project the crop prediction back into the original shape of the brain.
    """
    # Original shape of the volume
    x, y, z = full_brain_pred.shape[0], full_brain_pred.shape[1], full_brain_pred.shape[2]

    # Coordinates where the crop was extracted from the original volume
    cropx_i, cropy_i, cropz_i, cropx_e, cropy_e, cropz_e = crop_parameters

    startx = x//2-(cropx_i)
    starty = y//2-(cropy_i)
    startz = z//2-(cropz_i)
    
    endx = x//2+(cropx_e)
    endy = y//2+(cropy_e)
    endz = z//2+(cropz_e)

    full_brain_pred[startx:endx, starty:endy, startz:endz] = crop_pred

    return full_brain_pred
