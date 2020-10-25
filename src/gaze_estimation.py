import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import cv2
import math

class Model_Gaze_Est:
   
	def __init__(self, model_name, device, extensions=None):
		
		self.model_weights=model_name+'.bin'
		self.model_structure=model_name+'.xml'
		self.device=device
		self.extensions=extensions
	   
		try:
			self.core=IECore()
			self.network=IENetwork(model=self.model_structure, weights=self.model_weights)
			network_layers=self.network.layers.keys()
			supported_layers=self.core.query_network(network=self.network,device_name=self.device).keys()
			for layer in network_layers:
				if layer in supported_layers:
					pass
				else:
					ext_required= True
					break
		
			if self.extensions!=None and "CPU" in self.device and ext_required:
				self.core.add_extension(self.extensions, self.device)

				for layer in network_layers:
					if layer in supported_layers:
						pass
					else:
						msg = "Layer extension doesn't support all layers"
						log.error(msg)
						raise Exception(msg)

			
		except Exception as e:
			raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

		self.input_name=[i for i in self.network.inputs.keys()]
		self.input_shape=self.network.inputs[self.input_name[1]].shape
		self.output_name=next(iter(self.network.outputs))
		self.output_shape=self.network.outputs[self.output_name].shape

	def load_model(self):
		self.exec_network=self.core.load_network(self.network, self.device)
		return

	def predict(self,left_eye,right_eye,head_pose):
		input_left_eye,input_right_eye=self.preprocess_input(left_eye,right_eye)
		input_dict={'left_eye_image':input_left_eye,'right_eye_image':input_right_eye,'head_pose_angles':head_pose}
		outputs = self.exec_network.infer(input_dict)[self.output_name]
		
		return self.preprocess_outputs(outputs[0],head_pose[2])

	def preprocess_input(self, left_eye, right_eye):
		preprocessed_left_eye = cv2.resize(left_eye,(self.input_shape[3],self.input_shape[2]))
		preprocessed_left_eye = preprocessed_left_eye.transpose((2,0,1))

		preprocessed_right_eye = cv2.resize(right_eye,(self.input_shape[3],self.input_shape[2]))
		preprocessed_right_eye = preprocessed_right_eye.transpose((2,0,1))

		return preprocessed_left_eye.reshape(1,*preprocessed_left_eye.shape),preprocessed_right_eye.reshape(1,*preprocessed_right_eye.shape)

	def preprocess_outputs(self,outputs,roll):
		cos = math.cos(roll*math.pi/180.0)
		sin = math.sin(roll*math.pi/180.0)

		x = outputs[0] * cos + outputs[1] * sin
		y = outputs[0] * sin + outputs[1] * cos 

		
		return (x,y),outputs