import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import cv2
class Model_HeadPos:

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
					ext_required=True
					break
		
			if self.extensions!=None and "CPU" in self.device and ext_required:
				self.core.add_extension(self.extensions, self.device)

				for layer in network_layers:
					if layer in supported_layers:
						pass
					else:
						msg="Layer extension doesn't support all layers"
						log.error(msg)
						raise Exception(msg)

			
		except Exception as e:
			raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

		self.input_name=next(iter(self.network.inputs))
		self.input_shape=self.network.inputs[self.input_name].shape
		self.output_name=next(iter(self.network.outputs))
		self.output_shape=self.network.outputs[self.output_name].shape

	def load_model(self):
		self.exec_network=self.core.load_network(self.network, self.device)
		return

	def predict(self, image):
		
		outputs=self.exec_network.infer({self.input_name:self.preprocess_input(image)})
		coords=self.preprocess_outputs(outputs,(image.shape[1],image.shape[0]))
		
		return coords

	def preprocess_input(self, image):
		preprocess_img=cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
		preprocess_img=preprocess_img.transpose((2,0,1))
		return preprocess_img.reshape(1,*preprocess_img.shape)

	def preprocess_outputs(self,outputs,dim):
		_=[]		
		for output in outputs:
			_.append(outputs[output][0][0])
		return _