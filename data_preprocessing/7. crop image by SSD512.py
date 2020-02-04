# Object Detection using SSD512(Single Shot Multibox Detector)

# Libraries
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
import imageio
import numpy as np
import os
import cv2

# Dependencies
from ssd512.keras_ssd512 import ssd_512
from ssd512.keras_ssd_loss import SSDLoss

# Defining the width and height of the image
height = 512
width = 512

# Defining confidence threshold
confidence_threshold = 0.5

# Different Classes of objects in VOC dataset
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


K.clear_session() # Clear previous models from memory.

# creating model and loading pretrained weights
model = ssd_512(image_size=(height, width, 3),	# dimensions of the input images (fixed for SSD512)
                n_classes=20,	# Number of classes in VOC 2007 & 2012 dataset
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], 
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
               two_boxes_for_ar1=True,
               steps=[8, 16, 32, 64, 128, 256, 512],
               offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               normalize_coords=True,
               subtract_mean=[123, 117, 104],
               swap_channels=[2, 1, 0],
               confidence_thresh=0.5,
               iou_threshold=0.45,
               top_k=200,
               nms_max_output_size=400)

# path of the pre trained model weights 
weights_path = 'ssd512/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5'
model.load_weights(weights_path, by_name=True)

# Compiling the model 
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


# Transforming image size
def transform(input_image):
	return cv2.resize(input_image, (512, 512), interpolation = cv2.INTER_CUBIC)

# Function to detect objects in image
def detect_object(copy_image, dog_name, file):
	original_image_height, original_image_width = copy_image.shape[:2]
	input_image = transform(copy_image)
	input_image = np.reshape(input_image, (1, 512, 512, 3))
	y_pred = model.predict(input_image)
	actual_prediction = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

	for i,box in enumerate(actual_prediction[0]):
		# print(box)
		# Coordinates of diagonal points of bounding box
		# x0 = box[-4] * original_image_width / width
		# y0 = box[-3] * original_image_height / height
		# x1 = box[-2] * original_image_width / width
		# y1 = box[-1] * original_image_height / height
		left = box[-4] * original_image_width / width
		top = box[-3] * original_image_height / height
		right = box[-2] * original_image_width / width
		bottom = box[-1] * original_image_height / height
		detect_width = right - left
		detect_height = bottom - top

		#center
		x = (right - left)/2
		y = (bottom - top)/2

		if classes[int(box[0])]=='dog':
			if width >= height:
				left = int(x - detect_width / 2)
				top = int(y - detect_width / 2)
				right= int(x + detect_width / 2)
				bottom= int(y + detect_width / 2)
			else:
				left = int(x - detect_height / 2)
				top = int(y - detect_height / 2)
				right= int(x + detect_height / 2)
				bottom= int(y + detect_height / 2)

			left -= 100
			right += 100
			top -=100
			bottom +=100

			if left <0:
				margin = np.abs(left)
				left = 0
				top = int(top + margin/2)
				bottom = int(bottom - margin/2)


			if top <0:
				margin = np.abs(top)
				top = 0
				left = int(left + margin/2)
				right = int(right - margin/2)
				
			crop_image = copy_image[int(top):int(bottom), int(left):int(right)]
			crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
			
			file = file.replace('./jpeg','')
			save_path = f"{new_dir}/{dog_name}/{file}_{i}.jpeg"
			cv2.imwrite(save_path, crop_image)
				
# Detecting objects in images
base_path = '../data/'
old_dir = os.path.join(base_path, '5.merge(stanfold,identification,crawling)_120').replace('\\','/')
new_dir = '../data/6.image crop by YOLO3'

for root, dirs, files in os.walk(old_dir):
	dog_name = os.path.split(root)[1]
	
	# make dir
	new_dog_dir = os.path.join(new_dir, dog_name)
	try:
		os.mkdir(new_dog_dir)
	except Exception as err:
		print(err)
		
	# img crop and save	
	for file in files:		
		input_image_path = os.path.join(root, file)
		original_image = cv2.imread(input_image_path, 1)
		original_image= cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
		copy_image = original_image.copy()
		if original_image is not None:
			detect_object(copy_image, dog_name, file)	# detecting objects

		
# Detecting objects in video
# for file in os.listdir(input_video_path):
# 	print('Reading', file)
# 	video_reader = imageio.get_reader(os.path.join(input_video_path, file))	# Reading video
# 	fps = video_reader.get_meta_data()['fps']	# gettinf fps of the image
# 	video_writer = imageio.get_writer(os.path.join(output_video_path, file), fps = fps)	# Writing back output image
# 	for i, frame in enumerate(video_reader):
# 		output_frame = detect_object(frame)	# detecting objects frame by frame
# 		video_writer.append_data(output_frame)	# appending frame to vidoe
# 		print('frame ', i, 'done')
# 	video_writer.close()
	

# This section is for realtime object detection on any video or movie or through webcam or any camera.
# If you want to do, follow these steps
# 1. Uncomment this part.
# 2. Put the path of video or movie here. You can also use your webcam or other camera(if there).
# 3. for  webcam use just put 0 in path (without quotes)
# 4. for secondary camera, use1 in path.
# 5. Once the video started, you can see the realtime object detection
# 6. Press 'q' button on the keyboard to exit.
"""
video_capture = cv2.VideoCapture(0) #Put path in bracket here	
while video_capture.isOpened():
    _, frame = video_capture.read() 
    canvas = detect_object(frame)
    cv2.imshow('Video', canvas) 
    if cv2.waitKey(1) & 0xFF == ord('q'): # To stop the loop.
        break 

video_capture.release() # We turn the webcam/video off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.
"""