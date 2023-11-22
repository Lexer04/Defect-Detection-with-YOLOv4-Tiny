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
topic1 = "ps_get_result" # Photosensor
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
paling = 0
b = []
x = [1,2,3]
y = []
a = []

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
	# x_string = json.dumps(kelas)
	# now = datetime.now()
	# current_time = now.strftime("%H:%M:%S")
	# date_time = now.strftime("%m/%d/%Y, %H:%M:%S: ")
	send = str(msg)
	result = client.publish(topic2, send)
	status = result[0]
	if status == 0:
		print(f"Send '{send}' to topic '{topic2}'")
	else:
		print(f"failed to send to topic'{topic2}'")
	
def subscribe(client: mqtt_client): #jetson
	def on_message(client, userdata, msg):
		global x,b,a
		print(f"Received `{msg.payload.decode('utf-8')}` from `{msg.topic}` topic")
		b = msg.payload.decode('utf-8')
		# print(b)
		a.append(b)
		receivedArray = b.split(',')
	# print(b)
	client.subscribe(topic)
	client.on_message = on_message

def subscribe1(client: mqtt_client): #recieve eric
	def on_message(client, userdata, msg):
		global x, data
		print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
		data = msg.payload.decode()
		print(data)
		# print(data)
	client.subscribe(topic1)
	client.on_message = on_message


def run():
	client = connect_mqtt()
	client.loop_start()

if __name__ == '__main__':
	client = connect_mqtt()
	client.loop_start()
	
	# classesFile = "data/coco.names"
	classesFile = "data/obj.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Load image.
	cap = cv2.VideoCapture(0)
	cap.set(3,1280)
	cap.set(4,720)
	net = cv2.dnn.readNet(model='Top1Best.weights', config='cfg/Tissue/top.cfg')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
	# Process image.
	while True:
		
		ret, frame = cap.read()
		# cv2.resize(frame, (1280,720))
		# cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
		# cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)
		# frame1 = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
		detections = pre_process(frame, net)
		img = post_process(frame.copy(), detections)

		# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
		subscribe(client)
		
		if len(a) > 0:
			hitung = count_strings(a)
			maksimal=max(hitung.values())
			paling = [element for element, count in hitung.items() if count == maksimal]		#===================== Buat Jetson SIDE
			print(a)
			# print(paling)
			if len(a) > 2:
				subscribe1(client)
				print(data)
		# print(b)
		if len(x) > 0:
			counts = count_strings(x)
			max_count =max(counts.values())
			most_common = [element for element, count in counts.items() if count == max_count] #====================== Buat Jetson TOP

			

			print(counts)
			print(most_common)
			Flawless= x.count('Flawless')
			Wrinkled= x.count('Wrinkled')
			Shifted = x.count('Shifted')
			Tear = x.count("Tear")
			# print(Wrinkled)
			# print(Shifted)
			# print(Tear)
			# print(Flawless)
			print("most : {}".format(most_common))
			print(data)
			print(paling)
			if data =='true':
				time.sleep(0.2)
				subscribe1(client)
				print("truee")
				if most_common == ['Flawless']:
					print("flawwww")
					if Wrinkled > 0:
						# subscribe(client)
						print(paling)
						if paling == ['FlawlessSamping']:
							print("masok flales sampin")		#Kerut atas Tanpa cacat samping
							publish(client, "False, 13") 
							x = []
							paling =[]
						elif paling == ['WrinkledSamping']:
							print("masok flales sampin")
							publish(client, "False, 14")		#Kerut atas Kerut Samping
							x = []
							paling =[]
						elif paling == ['TearSamping']:
							print("masok flales sampin")
							publish(client, "False, 15")		#Kerut atas Robek Samping
							x = []
							paling =[]
						else:
							print("salaha kirim")
							x = []
							paling =[]
					elif Shifted > 0:
						# subscribe(client)
						print(b)
						if paling == ['FlawlessSamping']:
							print("masok flales sampin")
							publish(client, "False, 16")		#Label tidak rapih atas Tanpa cacat samping
							x = []
							paling =[]
						elif paling == ['WrinkledSamping']:
							print("masok flales sampin")
							publish(client, "False, 17")		#Label tidak rapih atas Kerut Samping
							x = []
							paling =[]
						elif paling == ['TearSamping']:
							print("masok flales sampin")
							publish(client, "False, 18")		#Label tidak rapih atas Robek Samping
							x = []
							b =[]
						else:
							print("salaha kirim")
							x = []
							paling =[]
					elif Tear > 0:
						# subscribe(client)
						print(b)
						if paling == ['FlawlessSamping']:
							print("masok flales sampin")
							publish(client, "False, 19")		#Robek atas Tanpa cacat Samping
							x = []
							paling =[]
						elif paling == ['WrinkledSamping']:
							print("masok flales sampin")
							publish(client, "False, 20")		#Robek atas Kerut Samping
							x = []
							paling =[]
						elif paling == ['TearSamping']:
							print("masok flales sampin")
							publish(client, "False, 21")		#Robek atas Robek samping
							x = []
							paling =[]
						else:
							print("salaha kirim")
							x = []
							paling =[]
					else:
						print(b)
						if paling == ['FlawlessSamping']:			#Tanpa cacat atas Tanpa Cacat Samping
							print("masok flales sampin")
							publish(client, "True, 11")
							x = []
							paling =[]
						elif paling == ['WrinkledSamping']:		#Tanpa cacat atas kerut samping
							print("masok flales sampin")
							publish(client, "False, 11")
							x = []
							paling =[]
						elif paling == ['TearSamping']:			#Tanpa cacat atas sobek samping
							print("masok flales sampin")
							publish(client, "False, 12")
							x = []
							paling =[]
						else:
							print("salaha kirim")
							x = []
							paling =[]
				elif most_common == ['Wrinkled']:
					subscribe(client)
					print(b)
					if paling == ['FlawlessSamping']:
						print("masok flales sampin")		#Kerut atas Tanpa cacat samping
						publish(client, "False, 13") 
						x = []
						paling =[]
					elif paling == ['WrinkledSamping']:
						print("masok flales sampin")
						publish(client, "False, 14")		#Kerut atas Kerut Samping
						x = []
						paling =[]
					elif paling == ['TearSamping']:
						print("masok flales sampin")
						publish(client, "False, 15")		#Kerut atas Robek Samping
						x = []
						paling =[]
					else:
						print("salaha kirim")
						x = []
						paling =[]
				elif most_common == ['Shifted']:
					subscribe(client)
					print(b)
					if paling == ['FlawlessSamping']:
						print("masok flales sampin")
						publish(client, "False, 16")		#Label tidak rapih atas Tanpa cacat samping
						x = []
						paling =[]
					elif paling == ['WrinkledSamping']:
						print("masok flales sampin")
						publish(client, "False, 17")		#Label tidak rapih atas Kerut Samping
						x = []
						paling =[]
					elif paling == ['TearSamping']:
						print("masok flales sampin")
						publish(client, "False, 18")		#Label tidak rapih atas Robek Samping
						x = []
						paling =[]
					else:
						print("salaha kirim")
						x = []
						paling =[]
				elif most_common == ['Tear']:
					subscribe(client)
					print(b)
					if paling == ['FlawlessSamping']:
						print("masok flales sampin")
						publish(client, "False, 19")		#Robek atas Tanpa cacat Samping
						x = []
						paling =[]
					elif paling == ['WrinkledSamping']:
						print("masok flales sampin")
						publish(client, "False, 20")		#Robek atas Kerut Samping
						x = []
						paling =[]
					elif paling == ['TearSamping']:
						print("masok flales sampin")
						publish(client, "False, 21")		#Robek atas Robek samping
						x = []
						paling =[]
					else:
						print("salaha kirim")
						x = []
						paling =[]
				# elif most_common == ['person']:
				# 	subscribe(client)
				# 	print(b)
				# 	if b == 'PersonSamping':
				# 		publish(client, "PersonAtasPersonSamping")
				# 		print("Sent to PLC as person")
				# 		x =[]
				# else:
				# 	x = []
				# 	b =[]
				# break
		cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		#img = cv2.resize(img, (1280,720))
		cv2.imshow('Output', img)	
		if cv2.waitKey(1) & 0xFF == ord('q'):	
			break
	cap.release()
	cv2.destroyAllWindows()
