# run single
#-------------------------

#change it to your images folder
pathImageFolder='/home/wzh/Documents/date_set/usb_cam/usb_cam_bag_3' 


#change it to the corresponding camera file.
cameraCalibFile='./cameras/usb_cam.txt' 


./../bin/test_dataset image="$pathImageFolder" calib="$cameraCalibFile" #start=xxx end=xxx times=xxx name=xxx

