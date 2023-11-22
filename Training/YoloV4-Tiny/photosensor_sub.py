import cv2
import numpy as np
from paho.mqtt import client as mqtt_client
import time
import random
from datetime import datetime
import json

#mqtt parameter
broker = '146.190.106.65'
port = 1883
topic = "detection/jetson" # Antar Jetson
topic1 = "ps_reserved" # Photosensor
topic2 = "status_product" # Kirim PLC	
client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'sorting'
password = 'fiHyn{odOmlap3@sorting'

# Constants.
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.5

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

# Contrast & Brightness
alpha = 0.7
beta = 10

global kelas,conf,x,x_string
kelas = 0
conf = 0
data = 0
b = []
x = []
y = []

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
	x_string = json.dumps(kelas)
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	date_time = now.strftime("%m/%d/%Y, %H:%M:%S: ")
	send = kelas
	result = client.publish(topic2, msg)
	status = result[0]
	
def subscribe(client: mqtt_client): #jetson
	def on_message(client, userdata, msg):
		global x,b
		print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
		b = msg.payload.decode()
		
		# receivedArray = b.split(',')
	client.subscribe(topic)
	client.on_message = on_message

def subscribe1(client: mqtt_client): #recieve eric
	def on_message(client, userdata, msg):
		global x, data
		print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
		data = msg.payload.decode()
		print(data)
	client.subscribe(topic1)
	client.on_message = on_message


def run():
	client = connect_mqtt()
	client.loop_start()

if __name__ == '__main__':
	client = connect_mqtt()
	client.loop_start()
	classesFile = "data/coco.names"
	# classesFile = "data/obj.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Load image.
	cap = cv2.VideoCapture(0)
	net = cv2.dnn.readNet(model='Top1Best.weights', config='cfg/Tissue/top.cfg')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
	# Process image.
	while True:
		subscribe(client)
		subscribe1(client)
		ret, frame = cap.read()
		frame1 = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
		detections = pre_process(frame1, net)
		img = post_process(frame1.copy(), detections)

		# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

		if len(x) > 0:
			counts = count_strings(x)
			max_count =max(counts.values())
			most_common = [element for element, count in counts.items() if count == max_count]
			print(counts)
			print(most_common)
			if data == 'true':
				
				if most_common == ['Flawless']:
						Wrinkled= x.count('Wrinkled')
						Shifted = x.count('Shifted')
						Tear = x.count("Tear")
						if Wrinkled > 0:
							subscribe(client)
							if b == 'FlawlesSamping':
								publish(client, "WrinkledAtasFlawlessSamping")
								x =[]
							elif b == 'WrinkledSamping':
								publish(client, "WrinkledAtasWrinkledSamping")
								x =[]
							elif b == 'TearSamping':
								publish(client, "WrinkledAtasTearSamping")
								x =[]
						elif Shifted > 0:
							subscribe(client)
							if b == 'FlawlesSamping':
								publish(client, "ShiftedAtasFlawlessSamping")
								x =[]
							elif b == 'WrinkledSamping':
								publish(client, "ShiftedAtasWrinkledSamping")
								x =[]
							elif b == 'TearSamping':
								publish(client, "ShiftedAtasTearSamping")
								x =[]
						elif Tear > 0:
							subscribe(client)
							if b == 'FlawlesSamping':
								publish(client, "TearAtasFlawlessSamping")
								x =[]
							elif b == 'WrinkledSamping':
								publish(client, "TearAtasWrinkledSamping")
								x =[]
							elif b == 'TearSamping':
								publish(client, "TearAtasTearSamping")
								x =[]
				elif most_common == ['Wrinkled']:
					subscribe(client)
					if b == 'FlawlesSamping':
						publish(client, "WrinkledAtasFlawlessSamping")
						x =[]
					elif b == 'WrinkledSamping':
						publish(client, "WrinkledAtasWrinkledSamping")
						x =[]
					elif b == 'TearSamping':
						publish(client, "WrinkledAtasTearSamping")
						x =[]
				elif most_common == ['Shifted']:
					subscribe(client)
					if b == 'FlawlesSamping':
						publish(client, "ShiftedAtasFlawlessSamping")
						x =[]
					elif b == 'WrinkledSamping':
						publish(client, "ShiftedAtasWrinkledSamping")
						x =[]
					elif b == 'TearSamping':
						publish(client, "ShiftedAtasTearSamping")
						x =[]
				elif most_common == ['Tear']:
					subscribe(client)
					if b == 'FlawlesSamping':
						publish(client, "TearAtasFlawlessSamping")
						x =[]
					elif b == 'WrinkledSamping':
						publish(client, "TearAtasWrinkledSamping")
						x =[]
					elif b == 'TearSamping':
						publish(client, "TearAtasTearSamping")
						x =[]
				elif most_common == ['person']:
					subscribe(client)
					print(b)
					if b == 'PersonSamping':
						publish(client, "PersonAtasPersonSamping")
						print("Sent to PLC as person")
						x =[]
		cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		cv2.resize(img, (1280,720))
		cv2.imshow('Output', img)	
		if cv2.waitKey(10) & 0xFF == ord('q'):	
			break
	cap.release()
	cv2.destroyAllWindows()