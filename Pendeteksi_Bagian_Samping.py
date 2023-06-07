import cv2
import numpy as np
from paho.mqtt import client as mqtt_client
import random
from datetime import datetime

#mqtt parameter
broker = '146.190.106.65'
port = 1883
topic = "detection/jetson" # Antar Jetson
topic1 = "ps_get_result" # Photosensor
topic2 = "status_product" # Kirim PLC	
client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'sorting'
password = 'fiHyn{odOmlap3@sorting'

# Constants.
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
SCORE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.9

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

#Variable
global kelas,conf,x,x_string
kelas = 0
conf = 0
data = 0
x = []
y = []
c = 0
Wrinkled = 0
Shifted = 0
Tear = 0

def count_strings(array):
	counts = {}
	for element in array:
		if element in counts:
			counts[element] += 1
		else:
			counts[element] = 1
	return counts

def draw_label(input_image, label, left, top):
	"""Draw text onto image at location."""
	# Get text size.
	text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
	dim, baseline = text_size[0], text_size[1]
	# Use text size to create a BLACK rectangle. 
	cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
	# Display text inside the rectangle.
	cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)

	return outputs


def post_process(input_image, outputs):
	# Lists to hold respective values while unwrapping.
	global kelas, conf, x, x_string
	class_ids = []
	confidences = []
	boxes = []

	image_height, image_width = input_image.shape[:2]

	# Resizing factor.
	x_factor = image_width
	y_factor =  image_height

	# Iterate through 25200 detections.
	for out in outputs:
		for detection in out:
			classes_scores = detection[5:]
			class_id = np.argmax(classes_scores)
			confidence = classes_scores[class_id]
			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = detection[0], detection[1], detection[2], detection[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
			  
				box = np.array([left, top, width, height])
				boxes.append(box)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	kelas = ''
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, 3)
		label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
		draw_label(input_image, label, left, top)
		kelas = classes[class_ids[i]]
		conf = confidences[i]
		x.append(kelas)
		y.append(conf)
	return input_image

def connect_mqtt():
	def on_connect(client, userdata, flags, rc):
		if rc == 0:
			print("Connected to MQTT Broker!")
		else:
			print("Failed to connect, return code %d\n", rc)
	# Set Connecting Client ID
	client = mqtt_client.Client(client_id)
	client.username_pw_set(username, password)
	client.on_connect = on_connect
	client.connect(broker, port)
	return client

def publish(client,msg):
	send = str(msg)
	result = client.publish(topic, send)
	status = result[0]
	if status == 0:
		print(f"Mengirim '{send}' ke topic '{topic}'")

def subscribe1(client: mqtt_client): #Photoelectric Sensor
	def on_message(client, userdata, msg):
		global x, data
		data = msg.payload.decode()
	client.subscribe(topic1)
	client.on_message = on_message

def run():
	client = connect_mqtt()
	client.loop_start()

if __name__ == '__main__':
	client = connect_mqtt()
	client.loop_start()
	classesFile = "data/obj.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Load image.
	# camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
	# cap = cv2.VideoCapture(camSet)
	cap = cv2.VideoCapture(0)
	cap.set(3,1280)
	cap.set(4,720)
	net = cv2.dnn.readNet(model='side_best_new.weights', config='cfg/Tissue/side.cfg')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
	# Process image.
	while True:
		subscribe1(client)
		ret, frame = cap.read()
		frame_flip = cv2.flip(frame,1)
		detections = pre_process(frame_flip, net)
		img = post_process(frame_flip.copy(), detections)
		# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

		if len(x) > 0:
			counts = count_strings(x)
			count2 = len(x)
			max_count =max(counts.values())
			most_common = [element for element, count in counts.items() if count == max_count]
			print(counts)
			Wrinkled= x.count('Wrinkled')
			Shifted = x.count('Shifted')
			Tear = x.count("Tear")
			Flawless = x.count("Flawless")
			if c == 0 and int(count2) > 0:
				if most_common == ['Flawless']:
					if Wrinkled < 0:
						publish(client, "WrinkledSamping")					
					elif Tear < 0:
						publish(client, "TearSamping")
					else:
						publish(client, "FlawlessSamping")
				elif most_common == ['Wrinkled']:
					publish(client, 'WrinkledSamping')
				elif most_common == ['Tear']:
					publish(client, 'TearSamping')
				elif Flawless == Wrinkled == Tear:
					publish(client, 'TearSamping')
				elif Flawless > 0:
					if Flawless == Wrinkled:
						publish(client, 'WrinkledSamping')
					elif Flawless == Tear:
						publish(client, 'TearSamping')
				elif Wrinkled > 0:
					if Wrinkled == Tear:
						publish(client, 'TearSamping')
					elif Wrinkled == Flawless:
						publish(client, 'WrinkledSamping')
				elif Tear > 0:
					if Tear == Flawless:
						publish(client, 'TearSamping')
					elif Tear == Wrinkled:
						publish(client, 'TearSamping')
				c = count2
			if (count2-c) > 0:
				if most_common == ['Flawless']:
					if Wrinkled > 0:
						publish(client, "WrinkledSamping")
					elif Tear > 0:
						publish(client, "TearSamping")
					else:
						publish(client, "FlawlessSamping")
				elif most_common == ['Wrinkled']:
					publish(client, 'WrinkledSamping')
				elif most_common == ['Tear']:
					publish(client, 'TearSamping')
				c = count2
			elif most_common == ['Wrinkled']:
				publish(client, 'WrinkledSamping')
			elif most_common == ['Tear']:
				publish(client, 'TearSamping')
			elif Flawless == Wrinkled == Tear:
				publish(client, 'TearSamping')
			elif Flawless > 0:
				if Flawless == Wrinkled:
					publish(client, 'WrinkledSamping')
				elif Flawless == Tear:
					publish(client, 'TearSamping')
			elif Wrinkled > 0:
				if Wrinkled == Tear:
					publish(client, 'TearSamping')
				elif Wrinkled == Flawless:
					publish(client, 'WrinkledSamping')
			elif Tear > 0:
				if Tear == Flawless:
					publish(client, 'TearSamping')
				elif Tear == Wrinkled:
					publish(client, 'TearSamping')
			if data =='true':
				c = 0
				x = []
				data= ''
		cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		cv2.imshow('Output', img)	
		if cv2.waitKey(1) & 0xFF == ord('q'):	
			break
	cap.release()
	cv2.destroyAllWindows()
