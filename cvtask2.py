import numpy as np
import os
import sys
import cv2
import csv
import torch
from PIL import Image
from pathlib import Path
import shapely as shape
import matplotlib.pyplot as plt


f = open("data.txt", "w") 
f.close()

# Enter path here
image_folder_path = r"C:\Users\pravi\OneDrive\Documents\Python\COMPUTER VISION\TASK 2\KITTI_Selection\KITTI_Selection\images" 
kitti_folder_loc = r"C:\Users\pravi\OneDrive\Documents\Python\COMPUTER VISION\TASK 2\KITTI_Selection\KITTI_Selection"

def read_data(folder_path):
    """Args: Folder path 
    Returns: data in the form of dictionares of calib and ground truth labels """
    folder_path = folder_path + "/"
    calib = {}
    labels = {}
    folders = ["calib", "labels"]
    for folder in folders:
        pth = folder_path + folder
        f = os.listdir(pth) # lists all the csv files in folder
        for file in f:                              # loops tyhrough all csv files
            pth = folder_path + folder + "/" + file  # complete csv file location
            arr = []
            if(folder == "labels"):
                file1 = open(pth)
                csvreader = csv.reader(file1)
                for row in csvreader:
                    # print(row)
                    arr.append("".join(row).split())
                labels[file[:6]] = arr
                continue
            if(folder == "calib"):
                arr = np.loadtxt(pth, delimiter=" ")
                # calib[file[:6]] = np.array(arr, dtype=np.float16)
                calib[file[:6]] = np.array(arr)
    return(labels, calib)

def detect_and_draw(image_id):
    """Args: Image path
    Detects cars in the image 
    Returns image mat with bounding box around the cars and array with corner coordinates of bounding rectangle"""
    # YOLO v5 library path
    yolov5_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'ultralytics_yolov5_master'
    sys.path.append(str(yolov5_dir))
    # Load YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    image_path = image_folder_path +"/"+ image_id + ".png"
    # Just makes sure that the image in RGB
    img = Image.open(image_path).convert("RGB")
    # model function used the pretrained yolo model to detect OBJECTS in the image
    results = model(img, size=640) # Image is resized to 640 x 640
    # Drawing bounding boxes around car
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    car_bounding_boxes = []
    for box in results.xyxy[0]:
        label = int(box[-1])
        if label == 2:  # here class value 2 is for cars..... we elimainte other detected objects
            xmin, ymin, xmax, ymax = map(int, box[:4])
            car_bounding_boxes.append([[xmin, ymin], [xmax, ymax]])
            cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # Display Images
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bbx_dict = {}
    bbx_dict[image_id] = [car_bounding_boxes]
    return(image_rgb, bbx_dict)

def iou(yolo_arr, gt_arr):
    '''Returns iou of two object areas
    Args: Two arrays each array has 4 coordinates; 2 corner pts of rect [[x1,y1], [x2,y2]]'''
    obj_label = shape.Polygon([(yolo_arr[0][0], yolo_arr[0][1]), (yolo_arr[1][0],yolo_arr[0][1]), (yolo_arr[1][0], yolo_arr[1][1]), (yolo_arr[0][0], yolo_arr[1][1])])
    obj_predicion = shape.Polygon([(gt_arr[0][0], gt_arr[0][1]), (gt_arr[1][0], gt_arr[0][1]), (gt_arr[1][0], gt_arr[1][1]), (gt_arr[0][0], gt_arr[1][1])])
    poly_union = obj_label.union(obj_predicion)
    poly_intersection = obj_label.intersection(obj_predicion)
    iou = int(poly_intersection.area) / int(poly_union.area)
    return(round(iou,3))

def augument_pixel(pixel_val_arr, image_id):
    """Args: Array of all pixel coordinates[[x1, y1], [x2, y2]]
    Returns: Augumented array [[x1, y1, 1], [x2, y2, 1]]"""
    global bbx_dict
    
    for pixel_val in pixel_val_arr:
        for coordinate in pixel_val:
            coordinate.append(1)
    bbx_dict[image_id].append(pixel_val_arr)
    return(np.array(pixel_val_arr))

def select_centre_coordinate(pixel_val_array):
    """Selects centre coordinate of the array
    Args: Array with augumented pixel coordinates
    Returns: Complate array with Centre bottom line middle coordinate for each bounding box"""
    final_array = []
    for pixel_val in pixel_val_array:
        centre_bottom = np.array([0,0,0])
        for coordinate in pixel_val:
            y = coordinate[1]
            centre_bottom += coordinate
            final_pt = centre_bottom // 2
        final_pt[1] = y
        final_array.append(final_pt)
    return(np.array(final_array))

def calculate(array_with_sigle_coordinate, callib_mat):
    """Multiplies the pixel coordinates with inverse callibration matrix
    Args: Array of single point per bounding box, Callibretion matrix
    Returns: Helper coordinates"""
    final_arr = []
    inv_calib_mat = np.linalg.inv(callib_mat)
    for coordinate in array_with_sigle_coordinate:
        final_arr.append(np.dot(inv_calib_mat, coordinate))
    return(final_arr)

def calculate_distance(helper_coordinates_arr):
    """Calculates distance from the helper coordinates
    Args: Helper coordinate array"""
    final_arr = []
    for coordinate in helper_coordinates_arr:
        m = 1.65 / coordinate[1]
        new_coordinate = np.array([coordinate[0] * m, 1.65, m])
        # Calculate eucledian distance
        distance = np.sqrt(np.sum(np.square(new_coordinate)))
        final_arr.append([distance])
    return(final_arr)


# List of all the images 
f = os.listdir(image_folder_path)
f = [i[0:6] for i in f]
not_matched_boxes = 0

gt_list_plot = []
calc_list_plot = []

if(__name__ == "__main__"):
    # Iterating through all the images
    for image_id in f:
        # read data and store
        (labels, calib) = read_data(kitti_folder_loc)
        # global bbx_dict
        (image, bbx_dict) = detect_and_draw(image_id)
        # Appending gt rectangle corners to the bbx dict
        arr = []
        for i in labels[image_id]:
            arr.append([[int(float(i[1])), int(float(i[2]))], [int(float(i[3])), int(float(i[4]))]])
        bbx_dict[image_id].append(arr)

        calib_mat = calib[image_id]
        rect_coord = bbx_dict[image_id][0][:]
        aug_mat = augument_pixel(rect_coord, image_id)
        centre_coordinates = select_centre_coordinate(aug_mat)

        # # Values after K inverse and coordinates values
        helper_coordinates = calculate(centre_coordinates, calib_mat)
    
        dist_arr = calculate_distance(helper_coordinates)

        # Appending Ground truth distances to bbx_dict
        arr = []
        for i in labels[image_id]:
            arr.append(float(i[5]))
        bbx_dict[image_id].append(arr)

        # Appending calculated distances to bbx_dict
        bbx_dict[image_id].append(dist_arr)

        


        # This loop is for matching bounding boxes
        for i in range(len(bbx_dict[image_id][1])):  # ground truths
            (gt_ind, yolo_ind, best_iou) = (-1,-1,0)
            for j in range(len(bbx_dict[image_id][0])): # yolo detctions // calculated
                iou_val = iou(bbx_dict[image_id][1][i], bbx_dict[image_id][0][j])
                # print(iou_val)
                if (iou_val > 0.4):
                        if(best_iou < iou_val):
                            best_iou = iou_val
                            (gt_ind, yolo_ind) = (i,j)

            if(gt_ind == -1):
                # writing data in txt file
                print("Box not matched")
                not_matched_boxes += 1
            else:
                # writing data in txt file
                gt_list_plot.append(bbx_dict[image_id][3][gt_ind])
                calc_list_plot.append(bbx_dict[image_id][4][yolo_ind][0])
            # Bounding box for ground truth // colour: blue
            cv2.rectangle(image, bbx_dict[image_id][1][i][0], bbx_dict[image_id][1][i][1], (255,0,0), 2)
            calc_dist = str(round(bbx_dict[image_id][4][yolo_ind][0], 2)) # calc dist value
            gt_dist = str(round(bbx_dict[image_id][3][gt_ind], 2)) # gt distance value
            
            if(gt_ind != -1):

                font = cv2.FONT_HERSHEY_PLAIN
                font_scale = 0.8
                font_thickness = 1
                disp = 10
                # First text block
                calc_text = calc_dist
                (text_width, text_height), baseline = cv2.getTextSize(calc_text, font, font_scale, font_thickness)
                text_org = bbx_dict[image_id][1][gt_ind][0][0:2]

                # Display Calculated distances // colour: blue // calculated
                box_coords = ((text_org[0] + 10  , text_org[1] - text_height - baseline - disp), (text_org[0] + text_width + 10, text_org[1] + baseline - disp))
                cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
                cv2.putText(image, calc_text, (text_org[0]+10, text_org[1] - disp), font, font_scale, (0, 140, 0), font_thickness)

                # Second text block
                gt_text = gt_dist
                (text_width, text_height), baseline = cv2.getTextSize(gt_text, font, font_scale, font_thickness)
                text_org = bbx_dict[image_id][1][gt_ind][1][0:2]

                # Display ground truth distances // colour: green // groundtruth
                box_coords = ((text_org[0] + 5, text_org[1] - text_height - baseline + disp - 12), (text_org[0] + text_width +5, text_org[1] + baseline + disp - 12))
                cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
                cv2.putText(image, gt_text, (text_org[0]+5, text_org[1] + disp -12), font, font_scale, (226, 3, 3), font_thickness)



        text = image_id
        save_directory = r"C:\Users\pravi\OneDrive\Documents\Python\COMPUTER VISION\TASK 2\result images"
        # Define the filename
        filename = os.path.join(save_directory, f"{image_id}.png")
            
        # Save the image to the specified file
        cv2.imwrite(filename, image)
        print(f"Saved {filename}")
        # Uncomment this if you are displaying the image using matplotlib and uncomment only above 2 lines     
        cv2.imshow(image_id, image)
        cv2.waitKey(0)
        plt.imshow(image)
        plt.show()
