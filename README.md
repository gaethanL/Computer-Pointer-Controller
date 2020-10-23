# Computer-Pointer-Controller

This project, use a gaze detection model to control the mouse pointer of a computer. The Gaze Estimation model is used to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project was build using the InferenceEngine API from Intel's OpenVino ToolKit. 
More precisely the gaze estimation model take 3 inputs: - The head pose
														- The left eye image
														- The right eye imag
These 3 inputs are collected with remedies of 3 models: - Face Detection
														- Head Pose Estimation
														- Facial Landmarks Detection.
In order to make this project work, a pipline was build to coordinate the flow of data from the input, and then amongst the different models and finally to the mouse controller

## Getting Started

### Prerequisites

### Install Intel® Distribution of OpenVINO™ toolkit

Please follow instructions provided by this [OpenVINO-guide](https://docs.openvinotoolkit.org/latest/) document.

### Download the project 

Clone the  provide [repository](https://github.com/gelhteag/Computer-Pointer-Controller)



### initialize openVINO Environment

For windows OS 

on a command prompt in administrator mode add the following  line

```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
```
then run 
```
setupvars.bat
```
### Download the pre-trainde models

 ⋅[Face Detection Model](https://docs.openvinotoolkit.org/2020.2/_models_intel_face_detection_retail_0005_description_face_detection_retail_0005.html)
 ⋅[Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
 ⋅[Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
 ⋅[Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

 #### Download the models by using openVINO model downloader
 ```
 python <openvino directory>/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-retail-0005"
```

```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"

```

## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
