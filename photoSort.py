import csv

from os import listdir
from os.path import isfile, join
import argparse
import sys
import time
import numpy as np
import tensorflow as tf
from scripts.label_image import read_tensor_from_image_file
from scripts.label_image import load_labels
from scripts.label_image import load_graph

filesPath = 'test/'

filesToSort = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]
model_file = "tf_files/retrained_graph.pb"
label_file = "tf_files/retrained_labels.txt"
input_height = 224 
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"
sortedFilesDict = {}
graph = load_graph(model_file)
finalPercent = 0.0
for f in sorted(filesToSort):
	print('Evaluating : {}'.format(f))
	file_name = filesPath+f
	t = read_tensor_from_image_file(file_name,
        	                          input_height=input_height,
                	                  input_width=input_width,
                        	          input_mean=input_mean,
                                	  input_std=input_std)

	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name);
	output_operation = graph.get_operation_by_name(output_name);
	with tf.Session(graph=graph) as sess:
		start = time.time()
    		results = sess.run(output_operation.outputs[0],
        		              {input_operation.outputs[0]: t})
   		end=time.time()
	results = np.squeeze(results)
	top_k = results.argsort()[-5:][::-1]
	labels = load_labels(label_file)
	print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
	for i in top_k:
		print(labels[i], results[i])
	sortedFilesDict[f] = labels[top_k[0]]
	finalPercent = finalPercent + results[top_k[0]]
finalPercent = finalPercent / float(len(filesToSort))
print("Final recognition confidence percent {}".format(finalPercent))

with open('result.csv', 'wb') as csvfile:
	resultCSV = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for key, value in sortedFilesDict.items():
   		resultCSV.writerow([key] + [value])


