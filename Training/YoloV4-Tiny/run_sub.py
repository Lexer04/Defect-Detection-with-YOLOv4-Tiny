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
topic1 = "ps_get_sorting_result" # Photosensor Tambahan
topic2 = "status_product" # Kirim PLC
topic3 = "ps_detect"	# Photosensor 1
topic4 = "speed" #kecepatan konveyor
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
sensor = 0
sensor2 = []
speed = 0
x = []
y = []
start = 0
data = []

# def count_flawless(array, string):
#     countflawless = 0
#     for element in array:
#         if element == string:
#             countflawless += 1
#     return countflawless

# def count_wrinkled(array, string):
#     countwrinkled = 0
#     for element in array:
#         if element == string:
#             countwrinkled += 1
#     return countwrinkled

# def count_shifted(array, string):
#     countshifted = 0
#     for element in array:
#         if element == string:
#             countshifted += 1
#     return countshifted

# def count_tear(array, string):
#     counttear = 0
#     for element in array:
#         if element == string:
#             counttear += 1
#     return counttear

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
	# print(outputs[0].shape)

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
	# x =[]
	# x.append(kelas)
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
		# x_string = json.dumps(x)
		# x = [confidences[i]]
		# x = []
		# x.append(conf)
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
	# global x_string
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	date_time = now.strftime("%m/%d/%Y, %H:%M:%S: ")
	# send = str(date_time)+str(msg)
	send = kelas
	result = client.publish(topic2, msg)
	# result: [0, 1]
	status = result[0]
	# if status == 0:
	# 	print(f"Send `{send}` to topic `{topic2}`")
	# else:
	# 	print(f"Failed to send message to topic {topic}")

# def subscribe(client: mqtt_client, topic):
#     def on_message(client, userdata, msg):
#         print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
#         b = msg.payload.decode()
#         x.append(b)
#         receivedArray = b.split(',')
#         # x.extend(receivedArray)
#         # print(x)
#         # print(f"Received `{msg.payload.decode()}` from `{msg.array}` topic")

#     client.subscribe(topic)
#     client.on_message = on_message

def subscribe(client: mqtt_client): #jetson
	def on_message(client, userdata, msg):
		global x
		print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
		b = msg.payload.decode()
		x.append(b)
		receivedArray = b.split(',')
		#return msg
		# x.extend(receivedArray)
		# print(x)
		# print(f"Received `{msg.payload.decode()}` from `{msg.array}` topic")

	client.subscribe(topic)
	client.on_message = on_message

# def subscribe1(client: mqtt_client): #Photosensor tambahan
# 	def on_message(client, userdata, msg):
# 		global x, data
# 		print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
# 		data = msg.payload.decode()
# 		# print(data)
		
	
# 	client.subscribe(topic1)
# 	client.on_message = on_message

# def subscribe2(client: mqtt_client): #Photosensor 1
# 	def on_message(client, userdata, msg):
# 		global sensor
# 		print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
# 		sensor = msg.payload.decode()
# 		print(sensor)
		
	
# 	client.subscribe(topic4)
# 	client.on_message = on_message


def subscribe2(client: mqtt_client): #Kecepatan Konveyor
	# global sensor, sensor2
	def on_message(client, userdata, msg):
		global sensor, sensor2
		print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
		sensor = msg.payload.decode()
		sensor2 = sensor
		# print(sensor)
		# print(msg.payload.decode())
	client.subscribe(topic3)
	client.on_message = on_message

	subscribe(client) 	#jetson
		# subscribe1(client)	#Photosensor Tambahan
	subscribe2(client)	#Photosensor 
	subscribe3(client)	#Kecepatan Konveyor
	print(sensor2)
	if sensor == 'true':
		print("setart")
		start_time = time.time()
		print(start_time)
		start = 1
		delay = speed /5 * 7.11
		if len(x) > 0:
			counts = count_strings(x)
			max_count =max(counts.values())
			most_common = [element for element, count in counts.items() if count == max_count]
			if start == 1:
				current_time = time.time()
				print(current_time)
				if current_time - start_time > 5: #dalam second
				# if speed == '400' :
				# publish(client, random.randint(0, 1000))
				# x =[]
				# print("hapus")
					if most_common == ['Flawless']:
						Wrinkled= x.count('Wrinkled')
						Shifted = x.count('Shifted')
						Tear = x.count("Tear")
					if Wrinkled > 0:
						publish(client, "false, 1")
					if Shifted > 0:
						publish(client, "false, 2")
					if Tear > 0:
						publish(client, "false, 3")
					else : 
						publish(client, "true, 1")
							# x =[]
						print("Sent to PLC as Flawless")
					if most_common == ['Wrinkled']:
							# most_common = 'false, 1'
						publish(client, 'false, 1')
							# x =[]
						print("Sent to PLC as Wrinkled")
					if most_common == ['Shifted']:
							# most_common = 'true, 3'
						publish(client, 'false, 2')
							# x =[]
						print("Sent to PLC as Shifted")
					if most_common == ['Tear']:
							# most_common = 'true, 4'
						publish(client, 'false, 3')
							# x =[]
						print("Sent to PLC as Tear")
					if most_common == ['person']:
							# mo = 'true, 5'
						publish(client, "false, 4")
							# print(most_common)
							# x =[]
						print("Sent to PLC as person")
					# time.sleep() 	################ NENTUIN WAKTU YO ##############

def subscribe3(client: mqtt_client): #Kecepatan Konveyor
	def on_message(client, userdata, msg):
		global speed
		print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
		speed = msg.payload.decode()
		# print(speed)
		
	
	client.subscribe(topic4)
	client.on_message = on_message

def run():
	client = connect_mqtt()
	client.loop_start()
	# while True:
	#     time.sleep(1)
	#     publish(client, random.randint(0, 1000))

if __name__ == '__main__':
	# Load class names.
	# run()
	client = connect_mqtt()
	client.loop_start()
	# client = connect_mqtt
	# client.loop_start()
	classesFile = "data/coco.names"
	# classesFile = "data/Tissue/top.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Load image.
	cap = cv2.VideoCapture(0)
	
	# Give the weight files to the model and load the network using them.
	#modelWeights = "yolov4_1_3_416_416_static.onnx"
 	#net = cv2.dnn.readNet(modelWeights)
	#net = cv2.dnn.readNet(model='../training/yolov4-tiny-custom_best.weights', config='cfg/yolov4-tiny-custom.cfg')
	# net = cv2.dnn.readNet(model='Side1Best.weights', config='cfg/yolov4-tiny-custom.cfg')
	# net = cv2.dnn.readNet(model='Top1Best.weights', config='cfg/Tissue/top.cfg')
	net = cv2.dnn.readNet(model='yolov4-tiny.weights', config='cfg/yolov4-tiny.cfg')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
	# Process image.
	while True:
		# global data
		
		
		# print(sensor2)
		# print("SENSOR JALAN ANJENG!")
		# print("halo")
		# print(sensor)
		# sensor2 = sensor
		# print(sensor2)
		# print(speed)
		# print(data)
		ret, frame = cap.read()
		# cv2.resize(frame, (1280,720))
		frame1 = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
		# frame1 = cv2.rotate(frame, cv2.ROTATE_180)		
		detections = pre_process(frame1, net)
		img = post_process(frame1.copy(), detections)

		# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
		# print(label)
		# print("class= " +str(kelas))
		# print("With Confidence = ",conf)
		# subscribe(client, 'detection/jetson')
		# countflawless = count_flawless(x, 'flawless')
		# countflawless = count_flawless(x, 'person')
		# countwrinkled = count_wrinkled(x, 'wrinkled')
		# countshifted = count_shifted(x, 'shifted')
		# counttear = count_tear(x, 'tear')
		# print(countflawless)
		subscribe3(client)	#Kecepatan Konveyor
		subscribe(client) 	#jetson
		# subscribe1(client)	#Photosensor Tambahan
		subscribe2(client)	#Photosensor 
		
		print(sensor2)
		# if sensor == 'true':
		# 		print("setart")
		# 		start_time = time.time()
		# 		print(start_time)
		# 		start = 1
		# 		delay = speed /5 * 7.11
		# 		if len(x) > 0:
		# 			counts = count_strings(x)
		# 			max_count =max(counts.values())
		# 			most_common = [element for element, count in counts.items() if count == max_count]
		# 			if start == 1:
		# 				current_time = time.time()
		# 				print(current_time)
		# 				if current_time - start_time > 5: #dalam second
		# 		# if speed == '400' :
		# 		# publish(client, random.randint(0, 1000))
		# 		# x =[]
		# 		# print("hapus")
		# 					if most_common == ['Flawless']:
		# 						Wrinkled= x.count('Wrinkled')
		# 						Shifted = x.count('Shifted')
		# 						Tear = x.count("Tear")
		# 					if Wrinkled > 0:
		# 						publish(client, "false, 1")
		# 					if Shifted > 0:
		# 						publish(client, "false, 2")
		# 					if Tear > 0:
		# 						publish(client, "false, 3")
		# 				else : 
		# 					publish(client, "true, 1")
		# 					# x =[]
		# 					print("Sent to PLC as Flawless")
		# 				if most_common == ['Wrinkled']:
		# 					# most_common = 'false, 1'
		# 					publish(client, 'false, 1')
		# 					# x =[]
		# 					print("Sent to PLC as Wrinkled")
		# 				if most_common == ['Shifted']:
		# 					# most_common = 'true, 3'
		# 					publish(client, 'false, 2')
		# 					# x =[]
		# 					print("Sent to PLC as Shifted")
		# 				if most_common == ['Tear']:
		# 					# most_common = 'true, 4'
		# 					publish(client, 'false, 3')
		# 					# x =[]
		# 					print("Sent to PLC as Tear")
		# 				if most_common == ['person']:
		# 					# mo = 'true, 5'
		# 					publish(client, "false, 4")
		# 					# print(most_common)
		# 					# x =[]
		# 					print("Sent to PLC as person")
		# 			time.sleep() 	################ NENTUIN WAKTU YO ##############
		# 		if speed == '600' :
		# 		# publish(client, random.randint(0, 1000))
		# 		# x =[]
		# 		# print("hapus")
		# 			if most_common == ['Flawless']:
		# 				Wrinkled= x.count('Wrinkled')
		# 				Shifted = x.count('Shifted')
		# 				Tear = x.count("Tear")
		# 				if Wrinkled > 0:
		# 					publish(client, "false, 1")
		# 				if Shifted > 0:
		# 					publish(client, "false, 2")
		# 				if Tear > 0:
		# 					publish(client, "false, 3")
		# 			else : 
		# 				publish(client, "true, 1")
		# 				# x =[]
		# 				print("Sent to PLC as Flawless")
		# 			if most_common == ['Wrinkled']:
		# 				# most_common = 'false, 1'
		# 				publish(client, 'false, 1')
		# 				# x =[]
		# 				print("Sent to PLC as Wrinkled")
		# 			if most_common == ['Shifted']:
		# 				# most_common = 'true, 3'
		# 				publish(client, 'false, 2')
		# 				# x =[]
		# 				print("Sent to PLC as Shifted")
		# 			if most_common == ['Tear']:
		# 				# most_common = 'true, 4'
		# 				publish(client, 'false, 3')
		# 				# x =[]
		# 				print("Sent to PLC as Tear")
		# 			if most_common == ['person']:
		# 				# mo = 'true, 5'
		# 				publish(client, "false, 4")
		# 				# print(most_common)
		# 				# x =[]
		# 				print("Sent to PLC as person")
		
			# print(x)
			# print(counts)
			# print(most_common)
			# if data == 'false':
			# 	# publish(client, random.randint(0, 1000))
			# 	x =[]
			# 	# print("hapus")
			# 	if most_common == ['Flawless']:
			# 		# most_common = 'true, 1'
			# 		publish(client, "true, 1")
			# 		# x =[]
			# 		print("Sent to PLC as Flawless")
			# 	if most_common == ['Wrinkled']:
			# 		# most_common = 'false, 1'
			# 		publish(client, 'false, 1')
			# 		# x =[]
			# 		print("Sent to PLC as Wrinkled")
			# 	if most_common == ['Shifted']:
			# 		# most_common = 'true, 3'
			# 		publish(client, 'false, 2')
			# 		# x =[]
			# 		print("Sent to PLC as Shifted")
			# 	if most_common == ['Tear']:
			# 		# most_common = 'true, 4'
			# 		publish(client, 'false, 3')
			# 		# x =[]
			# 		print("Sent to PLC as Tear")
			# 	if most_common == ['person']:
			# 		# mo = 'true, 5'
			# 		publish(client, "false, 4")
			# 		# print(most_common)
			# 		# x =[]
			# 		print("Sent to PLC as person")
			# # if most_common == 'person':
			# # 	print("flawless")
			# sensor2 = sensor
			# print("sensor : {}".format(sensor))
			# print("speed : " + str(speed))
			

			
			# 		time.sleep() 	################ NENTUIN WAKTU YO ##############
			# 	if speed == '800' :
			# 	# publish(client, random.randint(0, 1000))
			# 	# x =[]
			# 	# print("hapus")
			# 		if most_common == ['Flawless']:
			# 			Wrinkled= x.count('Wrinkled')
			# 			Shifted = x.count('Shifted')
			# 			Tear = x.count("Tear")
			# 			if Wrinkled > 0:
			# 				publish(client, "false, 1")
			# 			if Shifted > 0:
			# 				publish(client, "false, 2")
			# 			if Tear > 0:
			# 				publish(client, "false, 3")
			# 		else : 
			# 			publish(client, "true, 1")
			# 			# x =[]
			# 			print("Sent to PLC as Flawless")
			# 		if most_common == ['Wrinkled']:
			# 			# most_common = 'false, 1'
			# 			publish(client, 'false, 1')
			# 			# x =[]
			# 			print("Sent to PLC as Wrinkled")
			# 		if most_common == ['Shifted']:
			# 			# most_common = 'true, 3'
			# 			publish(client, 'false, 2')
			# 			# x =[]
			# 			print("Sent to PLC as Shifted")
			# 		if most_common == ['Tear']:
			# 			# most_common = 'true, 4'
			# 			publish(client, 'false, 3')
			# 			# x =[]
			# 			print("Sent to PLC as Tear")
			# 		if most_common == ['person']:
			# 			# mo = 'true, 5'
			# 			publish(client, "false, 4")
			# 			# print(most_common)
			# 			# x =[]
			# 			print("Sent to PLC as person")
			# 		time.sleep() 	################ NENTUIN WAKTU YO ##############
			# 	if speed == '1000' :
			# 	# publish(client, random.randint(0, 1000))
			# 	# x =[]
			# 	# print("hapus")
			# 		if most_common == ['Flawless']:
			# 			Wrinkled= x.count('Wrinkled')
			# 			Shifted = x.count('Shifted')
			# 			Tear = x.count("Tear")
			# 			if Wrinkled > 0:
			# 				publish(client, "false, 1")
			# 			if Shifted > 0:
			# 				publish(client, "false, 2")
			# 			if Tear > 0:
			# 				publish(client, "false, 3")
			# 		else : 
			# 			publish(client, "true, 1")
			# 			# x =[]
			# 			print("Sent to PLC as Flawless")
			# 		if most_common == ['Wrinkled']:
			# 			# most_common = 'false, 1'
			# 			publish(client, 'false, 1')
			# 			# x =[]
			# 			print("Sent to PLC as Wrinkled")
			# 		if most_common == ['Shifted']:
			# 			# most_common = 'true, 3'
			# 			publish(client, 'false, 2')
			# 			# x =[]
			# 			print("Sent to PLC as Shifted")
			# 		if most_common == ['Tear']:
			# 			# most_common = 'true, 4'
			# 			publish(client, 'false, 3')
			# 			# x =[]
			# 			print("Sent to PLC as Tear")
			# 		if most_common == ['person']:
			# 			# mo = 'true, 5'
			# 			publish(client, "false, 4")
			# 			# print(most_common)
			# 			# x =[]
			# 			print("Sent to PLC as person")
			# 		time.sleep() 	################ NENTUIN WAKTU YO ##############
			# 	if speed == '1200' :
			# 	# publish(client, random.randint(0, 1000))
			# 	# x =[]
			# 	# print("hapus")
			# 		if most_common == ['Flawless']:
			# 			Wrinkled= x.count('Wrinkled')
			# 			Shifted = x.count('Shifted')
			# 			Tear = x.count("Tear")
			# 			if Wrinkled > 0:
			# 				publish(client, "false, 1")
			# 			if Shifted > 0:
			# 				publish(client, "false, 2")
			# 			if Tear > 0:
			# 				publish(client, "false, 3")
			# 		else : 
			# 			publish(client, "true, 1")
			# 			# x =[]
			# 			print("Sent to PLC as Flawless")
			# 		if most_common == ['Wrinkled']:
			# 			# most_common = 'false, 1'
			# 			publish(client, 'false, 1')
			# 			# x =[]
			# 			print("Sent to PLC as Wrinkled")
			# 		if most_common == ['Shifted']:
			# 			# most_common = 'true, 3'
			# 			publish(client, 'false, 2')
			# 			# x =[]
			# 			print("Sent to PLC as Shifted")
			# 		if most_common == ['Tear']:
			# 			# most_common = 'true, 4'
			# 			publish(client, 'false, 3')
			# 			# x =[]
			# 			print("Sent to PLC as Tear")
			# 		if most_common == ['person']:
			# 			# mo = 'true, 5'
			# 			publish(client, "false, 4")
			# 			# print(most_common)
			# 			# x =[]
			# 			print("Sent to PLC as person")
			# 		time.sleep() 	################ NENTUIN WAKTU YO ##############
			# # if most_common == 'person':
			# # 	print("flawless")

		cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		cv2.resize(img, (1280,720))
		cv2.imshow('Output', img)	
		# randomvar = ['1', '2', '3', '4', '5']
		# jumlah3 = randomvar.count('3')
		# jumlah6 = randomvar.count('6')
		# print(jumlah3)
		# if jumlah3 > 0:
		# 	print("HALOOO")
		# else:
		# 	print(jumlah6)
		if cv2.waitKey(10) & 0xFF == ord('q'):	
			break
	cap.release()
	cv2.destroyAllWindows()