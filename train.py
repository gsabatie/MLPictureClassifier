import os


os.environ["ARCHITECTURE"] = "mobilenet_1.0_224"
os.system('python -m scripts.retrain --bottleneck_dir=tf_files/bottlenecks --how_many_training_steps=30000 --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}"  --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt  --architecture="${ARCHITECTURE}"   --image_dir=train --flip_left_right True --random_brightness=30')
