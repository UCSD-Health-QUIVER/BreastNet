from pydicom import dcmread
import SimpleITK as sitk
import numpy as np
import glob
import os
from pydicom.tag import Tag, BaseTag
import cv2
from multiprocessing import Pool



def poly2mask(vertex_row_coords: np.array, vertex_col_coords: np.array,
              shape: tuple) -> np.array:
    """[converts polygon coordinates to filled boolean mask]

    Args:
        vertex_row_coords (np.array): [row image coordinates]
        vertex_col_coords (np.array): [column image coordinates]
        shape (tuple): [image dimensions]

    Returns:
        [np.array]: [filled boolean polygon mask with vertices at
                     (row, col) coordinates]
    """
    xy_coords = np.array([vertex_col_coords, vertex_row_coords])
    coords = np.expand_dims(xy_coords.T, 0)
    mask = np.zeros(shape)
    cv2.fillPoly(mask, coords, 1)
    return np.array(mask, dtype=bool)


class ARTemis():
    def __init__(self, in_path:str):
        self.in_path = in_path
        self.out_path = os.path.join(os.path.dirname(in_path), "ARTemisPlots")
        self.session_num = os.path.basename(os.path.normpath(self.in_path))
        self.grouped_dicom = {}
        self.dvh_arrays = {}
        self.make_connections(in_path)
        self.max_dose = -1
        self.metrics = {}
    
    def make_connections(self, dicom_path:str)->None:
        '''
        This function will take a path to a folder containing dicom files and try to group them together based on the RTPlanLabel, RTPlanDescription, and SOPInstanceUID
        path: path to folder containing dicom files for one adaptive session
        '''


        plan_files = glob.glob(os.path.join(dicom_path, 'RTP*.dcm'))
        struct_files = glob.glob(os.path.join(dicom_path, 'RTS*.dcm'))
        dose_files = glob.glob(os.path.join(dicom_path, 'RTD*.dcm'))

        if len(plan_files) == 0:
            plan_files = glob.glob(os.path.join(dicom_path, 'RP*.dcm'))
            struct_files = glob.glob(os.path.join(dicom_path, 'RS*.dcm'))
            dose_files = glob.glob(os.path.join(dicom_path, 'RD*.dcm'))

        for plan_file in plan_files:
            ds = dcmread(plan_file, force=True)
            self.grouped_dicom[ds.RTPlanLabel] = {'plan_file': plan_file,'planDesc':"ds.RTPlanDescription",'planSOPInstanceUID':ds.SOPInstanceUID, 'struct_file': None, 'dose_file': None}
        for struct_file in struct_files:
            ds = dcmread(struct_file)
            for key in self.grouped_dicom.keys():
                if True:
                    self.grouped_dicom[key]['struct_file'] = struct_file
                    self.grouped_dicom[key]['structSOPInstanceUID'] = ds.SOPInstanceUID

        for dose_file in dose_files:
            ds = dcmread(dose_file)
            for key in self.grouped_dicom.keys():
                if ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID == self.grouped_dicom[key]['planSOPInstanceUID']:
                    self.grouped_dicom[key]["dose_file"] = dose_file
    
    def make_dose_array(self):
        '''
        This function will take the grouped dicom files and return a dictionary with the dose arrays for each plan
        '''
        #Param[0] = Dose Grid Image Position
        #Param[1] = Spacing
        #Param[2] = Shape of dose array

        for key in self.grouped_dicom.keys():
            dose_params = np.array([[0.,0.,0.],
                        [0.,0.,0.],
                        [0.,0.,0.]])
            dose_file = self.grouped_dicom[key]['dose_file']
            if dose_file is None:
                continue
            ds = dcmread(dose_file)
            dose_array = ds.pixel_array * float(ds.DoseGridScaling)
            dose_shape = dose_array.shape
            dose_params[0] = ds.ImagePositionPatient
            dose_params[1][0] = ds.PixelSpacing[0]
            dose_params[1][1] = ds.PixelSpacing[1]
            dose_params[1][2] = ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0]
            dose_params[2] = dose_shape

            dose_img = sitk.GetImageFromArray(dose_array)
            dose_img.SetSpacing((ds.PixelSpacing[0], ds.PixelSpacing[1], ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0]))
            dose_img.SetOrigin((ds.ImagePositionPatient[0], ds.ImagePositionPatient[1], ds.ImagePositionPatient[2]))
            self.grouped_dicom[key]['dose_image'] = dose_img
            self.grouped_dicom[key]['dose_params'] = dose_params
            self.grouped_dicom[key]['dose_array'] = dose_array
            self.max_dose = np.max(dose_array)

    def write_dose(self,out_dir:str)->None:
        for key in self.grouped_dicom.keys():
            dose_file = self.grouped_dicom[key]['dose_file']
            if dose_file is None:
                continue
            dose_img = self.grouped_dicom[key]['dose_image']

            sitk.WriteImage(dose_img, os.path.join(out_dir, key.replace("/","+") + '_dose.nii'))

    def write_masks(self, out_dir:str):
        for key in self.grouped_dicom.keys():
            for item_key in self.grouped_dicom[key].keys():
                if "mask_image" in item_key:
                    mask_img = self.grouped_dicom[key][item_key]
                    sitk.WriteImage(mask_img, os.path.join(out_dir, key.replace("/","+") + '_' + item_key.replace("mask_image","") + '_mask.nii'))


    def make_contours(self, contour_names: list[str])->None:
        if "lung l" not in contour_names:
            contour_names.append("lung l")
        if "tumorbed" not in contour_names:
            contour_names.append("tumorbed")
    
        for contour_name in contour_names:
            
            
            for key in self.grouped_dicom.keys():
                struct_file = self.grouped_dicom[key]['struct_file']
                if struct_file is None:
                    continue
                ds = dcmread(struct_file)

                dose_params = self.grouped_dicom[key]['dose_params'].astype(int)

                contourNames = ds.StructureSetROISequence
                idxContour = -1
                for idx, i in enumerate(contourNames, start=0):
                    if(contour_name in i.ROIName.lower()): 
                        idxContour = idx
                        break
                if idxContour == -1:
                    print(f"Could not find {contour_name} in the structure set!")
                    continue
                mask = np.zeros([dose_params[2][0], dose_params[2][1], dose_params[2][2]], dtype=np.uint8)
                if Tag((0x3006, 0x0039)) in ds.keys():
                    Contour_sequence = ds.ROIContourSequence[idxContour]

                    if Tag((0x3006, 0x0040)) in Contour_sequence:
                        Contour_data = Contour_sequence.ContourSequence
                        for i in range(len(Contour_data)):


                            matrix_points = self.reshape_contour_data(Contour_data[i].ContourData[:], key)
                            mask = self.return_mask(mask, matrix_points, geometric_type=Contour_data[i].ContourGeometricType, key=key)
                        mask = mask % 2
                    else:
                        print(f"This structure set had no data present for {contour_name}! Returning a blank mask")
                else:
                    print("This structure set had no data present! Returning a blank mask")
                spacing = self.grouped_dicom[key]['dose_image'].GetSpacing()
                origin = self.grouped_dicom[key]['dose_image'].GetOrigin() 
                mask_img = sitk.GetImageFromArray(mask)
                mask_img.SetSpacing(spacing)
                mask_img.SetOrigin(origin)
                self.grouped_dicom[key][contour_name + "_mask_image"] = mask_img
                self.grouped_dicom[key][contour_name + "_mask_array"] = mask
                

    def reshape_contour_data(self, as_array: np.array, plan_key:str) -> np.array:
        as_array = np.asarray(as_array)
        if as_array.shape[-1] != 3:
            as_array = np.reshape(as_array, [as_array.shape[0] // 3, 3])
        matrix_points = np.asarray([self.grouped_dicom[plan_key]['dose_image'].TransformPhysicalPointToIndex(as_array[i])
                                    for i in range(as_array.shape[0])])
        return matrix_points     
    
    def return_mask(self, mask: np.array, matrix_points: np.array, geometric_type: str, key: str) -> np.array:
        dose_params = self.grouped_dicom[key]['dose_params'].astype(int)

        col_val = matrix_points[:, 0]
        row_val = matrix_points[:, 1]
        z_vals = matrix_points[:, 2]
        if geometric_type != "OPEN_NONPLANAR":
            temp_mask = poly2mask(row_val, col_val, (dose_params[2][1], dose_params[2][2]))
      
            mask[z_vals[0], temp_mask] += 1
        else:
            for point_index in range(len(z_vals) - 1, 0, -1):
                z_start = z_vals[point_index]
                z_stop = z_vals[point_index - 1]
                z_dif = z_stop - z_start
                r_start = row_val[point_index]
                r_stop = row_val[point_index - 1]
                r_dif = r_stop - r_start
                c_start = col_val[point_index]
                c_stop = col_val[point_index - 1]
                c_dif = c_stop - c_start

                step = 1
                if z_dif != 0:
                    r_slope = r_dif / z_dif
                    c_slope = c_dif / z_dif
                    if z_dif < 0:
                        step = -1
                    for z_value in range(z_start, z_stop + step, step):
                        r_value = r_start + r_slope * (z_value - z_start)
                        c_value = c_start + c_slope * (z_value - z_start)
                        add_to_mask(mask=mask, z_value=z_value, r_value=r_value, c_value=c_value)
                if r_dif != 0:
                    c_slope = c_dif / r_dif
                    z_slope = z_dif / r_dif
                    if r_dif < 0:
                        step = -1
                    for r_value in range(r_start, r_stop + step, step):
                        c_value = c_start + c_slope * (r_value - r_start)
                        z_value = z_start + z_slope * (r_value - r_start)
                        add_to_mask(mask=mask, z_value=z_value, r_value=r_value, c_value=c_value)
                if c_dif != 0:
                    r_slope = r_dif / c_dif
                    z_slope = z_dif / c_dif
                    if c_dif < 0:
                        step = -1
                    for c_value in range(c_start, c_stop + step, step):
                        r_value = r_start + r_slope * (c_value - c_start)
                        z_value = z_start + z_slope * (c_value - c_start)
                        add_to_mask(mask=mask, z_value=z_value, r_value=r_value, c_value=c_value)
        return mask


    def print_dict(self):
        print(self.grouped_dicom)


    def getContourMaskArray(self, plan_key:str) -> tuple:
        '''
        concatenates all the masks for a given into one array
        returns tuple of (contourMaskArray, list of contour names)
        '''
        contour_names = []
        contourMaskArray = None
        for key in self.grouped_dicom[plan_key]:
            if "_mask_image" in key:
                contour_names.append(key.split("_mask_image")[0])
                if contourMaskArray is None:
                    contourMaskArray = np.expand_dims(sitk.GetArrayFromImage(self.grouped_dicom[plan_key][key]), axis = 0)
                else:
                    contourMaskArray = np.concatenate((contourMaskArray, np.expand_dims(sitk.GetArrayFromImage(self.grouped_dicom[plan_key][key]), axis = 0)), axis=0)
        return contourMaskArray, contour_names




    def getImageParam(self, imageFile:list):
        arrayParam = np.zeros([3,3], dtype = float)
        imageFile.sort(key = self.sorting) 
       
        data = dcm.read_file(imageFile[-1])

        arrayParam[1,:] = (data.PixelSpacing[0], data.PixelSpacing[1], data.SliceThickness)
        arrayParam[2,:] = (int(data.Rows), int(data.Columns), int(len(imageFile)))
        arrayParam[0,:] = data.ImagePositionPatient
   
        return arrayParam, data.ImageOrientationPatient
        
def make_session(path, contour_names):
    print(path)
    artemis = ARTemis(path)
    artemis.make_dose_array()
    artemis.make_contours(contour_names)
    return artemis
 

if __name__ == "__main__":

    import time
    start_time = time.time()
    args = []
    contour_names = []
    dicom_path = r'{your dicom path here}'
    for session in os.listdir(dicom_path):
        if "ARTemis" not in session:
            args.append((os.path.join(dicom_path, session),contour_names))
   
    with Pool(4) as p:
        list_arts = p.starmap(make_session, args)

    total_max_dose = np.max([x.max_dose for x in list_arts])
    args = [(artemis, total_max_dose) for artemis in list_arts]


    print("--- %s seconds ---" % (time.time() - start_time))

