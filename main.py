import os
import cv2
import hdf5storage
import nrrd
import argparse
from tqdm import tqdm

from utils.utils import *
from utils.config import cfg
from utils.train_transunet import TransUNetSeg
from utils.cactus import *


#---------------------
#     Parameters
#---------------------


# Full brain
FULL_CROP_PARAMETERS = [180, 200, 318, 180, 200, 62]

# 128^3 sub-volume in the top-right part of the volume
CROP_PARAM_TR = [139, -61, 149, -11, 189, -21] 

# 128^3 sub-volume in the top-left part of the volume
CROP_PARAM_TL = [150, 189, 139, -22, -61, -11]

# Segmentation Models
MODEL_PATH_TL = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/model_config_TL.pth')
MODEL_PATH_TR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/model_config_TR.pth')

# Classification Model
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/cactus.pth')

# Vignette dimensions
VIGNETTE_DIM = (32, 32) # (16, 16) # (64, 64)

# Number of channels
CHANNELS = 2 # MODIFY NUMBER OF CHANNELS

# Thresholds for binarization
SEUIL_ANOMALY_TL = 0.4
SEUIL_ANOMALY_TR = 0.3

SEUIL_ANOMALY = 0.56

# Size thresholds
MIN_LESION_SIZE = 50
LESION_PERCENTAGE = 0.9

# PC Intensity
PC_threshold_max = 60

# Select computing device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print('Cuda available :', device)
    



#---------------------------
#    Inference Functions
#---------------------------


def read_and_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_torch = cv2.resize(img, (cfg.transunet.img_dim, cfg.transunet.img_dim))
    img_torch = img_torch / 255.
    img_torch = img_torch.transpose((2, 0, 1))
    img_torch = np.expand_dims(img_torch, axis=0)
    img_torch = torch.from_numpy(img_torch.astype('float32')).to(device)

    return img, img_torch


def infer(volume, transunet, treshold=0.5, merged=True):

    preds = np.zeros(volume.shape, dtype=np.uint8)
    
    for i in tqdm(range(volume.shape[2])):
        # coronal_slice = volume[:, :, i]

        if i != 0 and i != (volume.shape[2]-1):
            coronal_slice = volume[:,:,i-1:i+2] # 3 consecutive slices
            # print(i, coronal_slice.shape, element_mask.shape)
        elif i == 0:
            coronal_slice = volume[:,:,:i+3] # 3 consecutive slices
            # print(i, coronal_slice.shape)
        else:
            coronal_slice = volume[:,:,i-2:]
            # print(i, coronal_slice.shape)

        img, img_torch = read_and_preprocess(coronal_slice)
        # print(img_torch.shape)
        with torch.no_grad():
            pred_mask = transunet.model(img_torch)
            # print(pred_mask)
            pred_mask = torch.sigmoid(pred_mask)
            # print(pred_mask)
            pred_mask = pred_mask.detach().cpu().numpy().transpose((0, 2, 3, 1))

        orig_h, orig_w = img.shape[:2]
        pred_mask = cv2.resize(pred_mask[0, ...], (orig_w, orig_h))
        pred_mask = thresh_func(pred_mask, thresh=treshold)

        preds[:, :, i] = pred_mask

    return preds

# python main.py -in ./data/Patient-26.mat -out ./output

if __name__ == "__main__":
    # Configuration de l'analyse des arguments
    parser = argparse.ArgumentParser(description="Script de prédiction des PWML en ETF 3D.")
    parser.add_argument('-in', '--input', type=str, required=True, help="Chemin vers le fichier d'entrée.")
    parser.add_argument('-out', '--output', type=str, required=True, help="Chemin vers le dossier de sortie.")
    
    args = parser.parse_args()

    #-------------------
    #    Load volume
    #-------------------


    # Load volumes
    image = hdf5storage.loadmat(args.input)['data_repcom']

    print('\nVol.shape :', image.shape)

    # Add padding before crop
    image = np.pad(image, ((100, 100), (100, 100), (100, 100)), 'constant')

    # Crop TR128 volume and mask to get the same shapes
    x_min, y_min, z_min, x_max, y_max, z_max = CROP_PARAM_TR
    image_TR = crop_center(image,x_min, y_min, z_min, x_max, y_max, z_max )

    # Crop TL128 volume and mask to get the same shapes
    x_min, y_min, z_min, x_max, y_max, z_max = CROP_PARAM_TL
    image_TL = crop_center(image,x_min, y_min, z_min, x_max, y_max, z_max )


    #-----------------------------------
    #     Make TransUNet prediction
    #-----------------------------------
    
    print(f'\n[STEP 1] Generating TransUNet prediction...')

    # Prediction TR
    transunet = TransUNetSeg(device)
    transunet.load_model(os.path.join(MODEL_PATH_TR))
    pred_mask_TR = infer(image_TR, transunet, treshold=SEUIL_ANOMALY_TR)
    TU_mask_TR = remove_small_lesions_threshold(pred_mask_TR, min_lesion_size=MIN_LESION_SIZE)

    # Prediction TL
    transunet = TransUNetSeg(device)
    transunet.load_model(os.path.join(MODEL_PATH_TL))
    pred_mask_TL = infer(image_TL, transunet, treshold=SEUIL_ANOMALY_TL)
    TU_mask_TL = remove_small_lesions_threshold(pred_mask_TL, min_lesion_size=MIN_LESION_SIZE)


    # # Save intermediate outputs
    # PRED_PATH = args.output
    # os.makedirs(PRED_PATH, exist_ok=True)

    # np.save(os.path.join(PRED_PATH, f'pred_TU_3S_TR128_t-{SEUIL_ANOMALY_TR}.npy'), pred_mask_TR.astype(np.uint8))
    # np.save(os.path.join(PRED_PATH, f'pred_TU_3S_TL128_t-{SEUIL_ANOMALY_TL}.npy'), pred_mask_TL.astype(np.uint8))
    # print(f'\n[INFO] Predictions from TransUNet saved here : {PRED_PATH}')


    #---------------------------
    #     Load CACTUS model
    #---------------------------
    

    # Load crossvit multiview (axial 2 et coronal 2)
    model = CrossViTMultiview(
                            image_size = 32,
                            channels = 3,
                            num_classes = 2,
                            patch_size_small = 2,
                            patch_size_large = 2, 
                            small_dim = 24, 
                            large_dim = 24, 
                            small_depth = 1, 
                            large_depth = 1
                )
    checkpoint = torch.load(MODEL_PATH, map_location=device)['model_state_dict']
    model.load_state_dict(checkpoint)

    print(f'\n[STEP 2] Removing false alarms with CACTUS...')

    #-----------------------------------------------------
    #    Connected Component Extraction & TR Prediction
    #-----------------------------------------------------


    TU_mask_correction_TR = TU_mask_TR.copy()
    volume = image_TR

    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(TU_mask_TR, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesions found in TR.')
    # else:
    #     print(N, 'lesions found.')

    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # List of centroids per lesion cluster with segid as key
    centroids = dict(enumerate(stats['centroids']))
    # print(lesions)
    # First element is always the background
    del centroids[0]

    for k in range(1, len(lesions)+1):
        # print(lesions[k], centroids[k])
        if lesions[k] > 0:
            vignetteA = extract_vignette(volume[int(centroids[k][0]), :, :], [int(centroids[k][1]), int(centroids[k][2])], dimensions=VIGNETTE_DIM)
            # vignetteS = extract_vignette(volume[:, int(centroids[k][1]), :], [int(centroids[k][0]), int(centroids[k][2])], dimensions=VIGNETTE_DIM)
            vignetteC = extract_vignette(volume[:, :, int(centroids[k][2])], [int(centroids[k][0]), int(centroids[k][1])], dimensions=VIGNETTE_DIM)
            
            # Make label prediction with the classification network
            vignette_pytorch = read_and_preprocess_multiview(vignetteA, vignetteC)

            with torch.no_grad():
                #model to eval mode
                model.eval()
                y_pred = model(vignette_pytorch)
                y_pred_tag = torch.softmax(y_pred, dim = 1)
                _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
                y_pred_tags = y_pred_tag[0][1] > SEUIL_ANOMALY

                if y_pred_tags == 0:
                    # print('FP')
                    # Modify the mask predicted with TransU-Net
                    lesion_cluster = np.where(output == k)
                    TU_mask_correction_TR[lesion_cluster] = 0

    TU_mask_correction_TR = remove_small_lesions_threshold(TU_mask_correction_TR, min_lesion_size=200)
    TU_mask_correction_TR = filter_with_PC_intensities(volume, TU_mask_correction_TR, PC_threshold_max, pred=True)



    #------------------------------------------------------
    #    Connected Component Extraction & TL Prediction
    #------------------------------------------------------


    TU_mask_correction_TL = TU_mask_TL.copy()
    volume = image_TL

    # apply connected component analysis to the thresholded image
    connectivity = 26 # 3D
    output, N = cc3d.connected_components(TU_mask_TL, connectivity=connectivity, return_N=True)
    # If there is no lesion
    if N == 0:
        print('No lesions found in TL.')
    # else:
    #     print(N, 'lesions found.')

    # Get number of voxels, bounding box and centroid for each lesion
    stats = cc3d.statistics(output)

    # Number of voxels per lesion cluster with segid as key
    lesions = dict(enumerate(stats['voxel_counts']))
    # print(lesions)
    # First element is always the background
    del lesions[0]

    # List of centroids per lesion cluster with segid as key
    centroids = dict(enumerate(stats['centroids']))
    # print(lesions)
    # First element is always the background
    del centroids[0]

    for k in range(1, len(lesions)+1):
        # print(lesions[k], centroids[k])
        if lesions[k] > 0:
            vignetteA = extract_vignette(volume[int(centroids[k][0]), :, :], [int(centroids[k][1]), int(centroids[k][2])], dimensions=VIGNETTE_DIM)
            # vignetteS = extract_vignette(volume[:, int(centroids[k][1]), :], [int(centroids[k][0]), int(centroids[k][2])], dimensions=VIGNETTE_DIM)
            vignetteC = extract_vignette(volume[:, :, int(centroids[k][2])], [int(centroids[k][0]), int(centroids[k][1])], dimensions=VIGNETTE_DIM)
            
            # Make label prediction with the classification network
            vignette_pytorch = read_and_preprocess_multiview(vignetteA, vignetteC)

            with torch.no_grad():
                #model to eval mode
                model.eval()
                y_pred = model(vignette_pytorch)
                y_pred_tag = torch.softmax(y_pred, dim = 1)
                _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
                y_pred_tags = y_pred_tag[0][1] > SEUIL_ANOMALY

                if y_pred_tags == 0:
                    # print('FP')
                    # Modify the mask predicted with TransU-Net
                    lesion_cluster = np.where(output == k)
                    TU_mask_correction_TL[lesion_cluster] = 0

    TU_mask_correction_TL = remove_small_lesions_threshold(TU_mask_correction_TL, min_lesion_size=200)
    TU_mask_correction_TL = filter_with_PC_intensities(volume, TU_mask_correction_TL, PC_threshold_max, pred=True)


    #---------------------------------
    #    Export for visualization
    #---------------------------------

    print(f'\n[STEP 3] Generating final output...')

    # Generate full brain mask
    pred_full_brain = np.zeros(image.shape, dtype=np.uint8)

    # Add TR prediction
    pred_full_brain = insert_mask(pred_full_brain, TU_mask_correction_TR, CROP_PARAM_TR)
    # Add TL prediction
    pred_full_brain = insert_mask(pred_full_brain, TU_mask_correction_TL, CROP_PARAM_TL)

    # Crop all volumes to reduce all patients to the same shapes
    x_min, y_min, z_min, x_max, y_max, z_max = FULL_CROP_PARAMETERS
    image = crop_center_full(image, FULL_CROP_PARAMETERS)
    pred_full_brain = crop_center_full(pred_full_brain, FULL_CROP_PARAMETERS)

    # Transformations for MITK visualisation
    image_MITK_full_brain = np.flip(np.flip(np.transpose(image, (1, 2, 0)), 2), 0)
    pred_MITK_full_brain = np.flip(np.flip(np.transpose(pred_full_brain, (1, 2, 0)), 2), 0)

    # For medical data, a patient-based left-handed coordinate frame,
    # with ordered basis vectors pointing towards left, anterior, and superior, respectively.
    header = {'space': 'left-anterior-superior', 'space directions': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}

    # Save for later MITK visualization
    new_dir = args.output
    # print(new_dir)
    os.makedirs(new_dir, exist_ok=True)
    nrrd.write(os.path.join(new_dir,'full_brain_image.nrrd'), image_MITK_full_brain, header)
    nrrd.write(os.path.join(new_dir,'full_brain_prediction.nrrd'), pred_MITK_full_brain, header) 
    print('\nALL DONE ! :)')