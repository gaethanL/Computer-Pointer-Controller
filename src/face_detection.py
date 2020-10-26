import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import cv2
class Model_Face_Detect:

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

	def predict(self, image,threshold):
		input_img=self.preprocess_input(image)
		input_dict={self.input_name:input_img}
		outputs=self.exec_network.infer(input_dict)[self.output_name]
		coords=self.preprocess_outputs(outputs,threshold,(image.shape[1],image.shape[0]))
		return self.face_coord(coords,image)

	def face_coord(self, coords, image):
		if(len(coords)==1):
			coords=coords[0]
			face_coords=image[coords[1]:coords[3], coords[0]:coords[2]]
			return coords,face_coords
		return False,False

	def preprocess_input(self, image):
		preprocessed_frame=cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
		preprocessed_frame=preprocessed_frame.transpose((2,0,1))
		return preprocessed_frame.reshape(1,*preprocessed_frame.shape)

	def preprocess_outputs(self,outputs,threshold,dim):
		_=[]
		for box in outputs[0][0]:
			ct=box[2]
			if ct > threshold:
				xmin=int(box[3]*dim[0])
				ymin=int(box[4]*dim[1])
				xmax=int(box[5]*dim[0])
				ymax=int(box[6]*dim[1])
				_.append([xmin,ymin,xmax,ymax])
		return _