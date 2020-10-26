#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import cv2
import numpy as np
import logging as log
from argparse import ArgumentParser


from face_detection import Model_Face_Detect
from facial_landmarks_detection import Model_Facial_Land
from head_pose_estimation import Model_HeadPos
from gaze_estimation import Model_Gaze_Est

from input_feeder import InputFeeder
from mouse_controller import MouseController

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser=ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to a face detection xml file with a trained model.")
    parser.add_argument("-hp", "--head_pose_estimation_model", required=True, type=str,
                        help="Path to a head pose estimation xml file with a trained model.")
    parser.add_argument("-fld", "--facial_landmark_detection_model", required=True, type=str,
                        help="Path to a facial landmark detection xml file with a trained model.")
    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation xml file with a trained model.")                    
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-ms_prec", "--mouse_precision", required=False, type=str,default="medium",
                        help="mouse precision needed values are high-low-medium")
    parser.add_argument("-ms_speed", "--mouse_speed", required=False, type=str,default="fast",
                        help="mouse speed needed values are fast-slow-medium")
    parser.add_argument("-disp", "--display_type", required=False, type=str,
                        help="single image mode yes/no", default="cam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")        
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def infer_on_stream(args):

    
    prob_threshold=args.prob_threshold
   

    face_detector_path=args.face_detection_model
    facial_landmark_path=args.facial_landmark_detection_model
    head_pose_path=args.head_pose_estimation_model
    gaze_est_path=args.gaze_estimation_model
    input_display=args.display_type

    device=args.device
    extension=args.cpu_extension
    input_file=args.input

    speed=args.mouse_speed
    precision=args.mouse_precision
   

    
   
    face_detector=Model_Face_Detect(model_name=face_detector_path,device=device,extensions=extension)
    log.info("face_detector object intitialised")
    face_landmark_detector=Model_Facial_Land(model_name=facial_landmark_path,device=device,extensions=extension)
    log.info("face_landmark_detector object initialised")
    head_pose_estimation=Model_HeadPos(model_name=head_pose_path,device=device,extensions=extension)
    log.info("head_pose_estimation object initialised")
    gaze_estimation=Model_Gaze_Est(model_name=gaze_est_path,device=device,extensions=extension)
    log.info("gaze_estimation object initialised")
    
    model_loading=time.time()

    start_time=time.time()
    face_detector.load_model()
    log.info("Face Detector Model Loaded...")
    face_landmark_detector.load_model()
    log.info("Facial Landmark Model Loaded...")
    head_pose_estimation.load_model()
    log.info("Head Pose Estimation Model Loaded...")
    gaze_estimation.load_model()
    log.info("Gaze Estimation Model Loaded...")
    total_models_load_time=time.time()-start_time 
  

    
    try:
        input_feeder=InputFeeder(input_display,input_file)
        input_feeder.load_data()
    except:
        log.error("Something went wrong with loading camera/mouse")
        exit(0)

    mouse=MouseController(precision,speed)
    frames=0

    start_inf_time=time.time()
    for ret,frame in input_feeder.next_batch():
        if not ret:
            break
        frames+=1

        key=cv2.waitKey(60)

        start_inf_disp=time.time()
        
        #original = "original"
        #cv2.namedWindow(original)        # Create a named window
        #cv2.moveWindow(original, 600,200)  # Move it to (40,30)
        #cv2.imshow(original,cv2.resize(frame,(600,600)))
        # Start inference on face_detection model
        face_coords,face_image=face_detector.predict(frame,prob_threshold)
       
        if (face_coords):

            # Start inference on face_landmarks_detection model
            eye_coords, left_eye,right_eye, image_proccess=face_landmark_detector.predict(face_image)
            
            # Start inference on head pose estimation model
            head_pose_angles=head_pose_estimation.predict(face_image)

            # Start inference on gaze estimation model
            mouse_coord,gaze_coord=gaze_estimation.predict(left_eye,right_eye,head_pose_angles)

            left_eye=(eye_coords[0][0]+15,eye_coords[0][1]+15)
            right_eye=(eye_coords[1][0]+15,eye_coords[1][1]+15)

            gaze_x=int(gaze_coord[0]*250)
            gaze_y=int(-gaze_coord[1]*250)

         
            cv2.arrowedLine(image_proccess, left_eye, (left_eye[0]+gaze_x,left_eye[1]+gaze_y),(80, 15, 120), 3)
            cv2.arrowedLine(image_proccess, right_eye, (right_eye[0]+gaze_x,right_eye[1]+gaze_y),(80, 15, 120), 3)
            
            inference_time=time.time()-start_inf_disp

            inf_time_display="Inference Time Per Frame: {:.3f}ms"\
                                .format(inference_time*1000)

            cv2.putText(image_proccess,inf_time_display, (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 250, 0), 1)

            infer_img="process_img"
            cv2.namedWindow(infer_img)        
            cv2.moveWindow(infer_img, 10,200)  # Move it to (10,200)
            cv2.imshow(infer_img,cv2.resize(image_proccess,(600,600)))
          
            mouse.move(mouse_coord[0],mouse_coord[1])
    
    total_inference_time=time.time()-start_inf_time
    fps=int(frames)/(total_inference_time)
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'stats_'+str(device)+'.txt'), 'w') as f:
       f.write("Inference Time: {:.3f}"\
                                .format(total_inference_time)+'\n')
       f.write("FPS:    {:.3f}"\
                                .format(fps)+'\n')
       f.write("Model Loading Time:  {:.3f}"\
                                .format(total_models_load_time)+'\n')

    input_feeder.close()
    cv2.destroyAllWindows()
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args=build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)


if __name__=='__main__':
    main()


