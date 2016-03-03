caffe_root = '../../caffe-future/'  # this file is expected to be in {caffe_root}/examples
import sys
import scipy.io as sio
sys.path.insert(0, caffe_root + 'python')

import numpy as np
from PIL import Image

import caffe

# load net
net = caffe.Net('deploy.prototxt', 'train_iter_20000.caffemodel', caffe.TEST)


test_flag = False
if test_flag:
	print('************ Start to test the trained model using an example ************')
	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	im = Image.open('coast_bea3.jpg')
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793,116.66876762,122.67891434))
	in_ = in_.transpose((2,0,1))
	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_
	# run net and take argmax for prediction
	net.forward()
	out = net.blobs['score'].data[0].argmax(axis=0)
	sio.savemat('result.mat', {'result':out})
	print('output max: %d; min: %d', out.max(), out.min());

# net surgery
# output the  parameters of each layer
print('************ Start to check the trained model ************')
output_para_flag = True
if output_para_flag:
  net_layer_list = net.params.keys()
  for i in range(len(net_layer_list)):
	temp_filter = net.params[net_layer_list[i]][0].data
	temp_bias = net.params[net_layer_list[i]][1].data
	if temp_filter.max() == temp_filter.min()and temp_filter.max() == 0:
		print("Warning : all of paramters in {} layer are zero!".format(net_layer_list[i]))
	else:
		print('Normal status in {} layer'.format(net_layer_list[i]))







