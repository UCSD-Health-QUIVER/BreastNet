import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
import torchio as tio

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def makeGlowingMask(input_image:sitk.Image,spacing:float=1.0, interpolation:str='gaussian',mesa=False) -> sitk.Image:
    """
    This function takes in a sitk image of a contour and returns a glowing mask
    input_image: sitk.Image
        This is the image of the contour
    spacing: float
        This is the spacing of the image
    interpolation: str
        This is the interpolation method used to upsample the image
    mesa: bool
        This is a flag to indicate if the output should be a 'msea' mask or not

    """

    # Read the image
    if type(input_image) == str:
        image = sitk.ReadImage(input_image)
        image_array = sitk.GetArrayFromImage(image)
    else:
        image = input_image
        image_array = sitk.GetArrayFromImage(image)
    
    # these transformations are used to upsample and downsample the image
    # this is done for speed and memory reasons
    Transformation = tio.Resample(8,image_interpolation='nearest')
    Up = tio.Resample(2., image_interpolation='gaussian')

    image_small_img = Transformation(image)
    image_small = Transformation(image)
    image_small = sitk.GetArrayFromImage(image_small)
 
    # indicies are used to find the distances between the non-zero voxels and the current voxel
    nz = np.nonzero(image_small)
    non_zero_indicies = np.array(list(zip(nz[0], nz[1], nz[2]))).astype(np.int8)
    non_zero3 = np.stack([non_zero_indicies for _ in range(32**3)], axis=1).astype(np.int8)


    slice_idx = []
    for i in range(0,32):
        for j in range(0,32):
            for k in range(0,32):
                slice_idx.append([i,j,k])
    slice_idx = np.array(slice_idx).astype(np.int8)

    large = np.abs(non_zero3 - slice_idx).astype(np.uint8)

    summed = np.sum(large, axis=2) + 1
  
    inverse = np.square(1/summed)
    isl = np.sum(inverse, axis=0).reshape((32,32,32))
    isl = sitk.GetImageFromArray(isl)
    isl.SetSpacing(image_small_img.GetSpacing())
    isl.SetOrigin(image_small_img.GetOrigin())
    isl.SetDirection(image_small_img.GetDirection())
    isl = Up(isl)
 
    isl = sitk.GetArrayFromImage(isl)
  

    # this is done to make the mask == 1
    # inside the contour
    if mesa:
        isl = np.where(image_array == 1, 0, isl)


        isl/=np.max(isl)
        isl = np.where(image_array == 1,1,isl)
    else:
        isl /= np.max(isl)
    isl = sitk.GetImageFromArray(isl)
    isl.SetSpacing(image.GetSpacing())
    isl.SetOrigin(image.GetOrigin())
    isl.SetDirection(image.GetDirection())

    return isl
