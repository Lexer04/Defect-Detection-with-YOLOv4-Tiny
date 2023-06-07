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
GREEN = (21,71,52)

# Variables
global kelas,conf,x,x_string
kelas = 0
conf = 0
data = []
paling = 0
b = []
x = []
y = []
a = []
d = 0
C = 0
F = 0
W = 0
T = 0

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
		if len(x) == 0:
			print("Melakukan deteksi pada bagian atas")
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
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	send = str(msg)
	result = client.publish(topic2, send)
	status = result[0]
	if status == 0:
		print(f"Mengirim '{send}' ke topic '{topic2}' pada waktu {current_time}")
	else:
		print(f"failed to send to topic'{topic2}'")
	
def subscribe(client: mqtt_client): #Receive from Pendeteksi Bagian Samping 
	def on_message(client, userdata, msg):
		global x,b,a, data,C
		b = msg.payload.decode('utf-8')
		if b == 'FlawlessSamping':
			a.append(b)
		elif b == 'WrinkledSamping':
			a.append(b)
		elif b == 'TearSamping':
			a.append(b)	
		elif b == 'true':
			data.append(b)
		elif b == 'false':
			data = []
			C = 0
		receivedArray = b.split(',')
	client.subscribe(topic)
	client.on_message = on_message

def subscribe1(client: mqtt_client): #recieve from photoelectric sensor
	def on_message(client, userdata, msg):
		global x, data, a
		b = msg.payload.decode()
		if a == []:
			client.unsubscribe(topic)
	client.subscribe(topic1)
	client.on_message = on_message

def unsubscribe(mqtt_client):
	client.unsubscribe(topic1)

def unsubscribe1(client: mqtt_client):
	client.unsubscribe(topic)

def run():
	client = connect_mqtt()
	client.loop_start()

def true11():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Flawless Atas dan Flawless Samping")
	publish(client, "true, 11")			#Flawless Atas Flawless Samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	hitung = 0
	maksimal = 0
	C = 0
	paling = []
	data = []
	d=0
def false12():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Flawless Atas dan Wrinkled Samping")
	publish(client, "false, 12")		#Flawless Atas Kerut Samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false13():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Flawless Atas dan Tear Samping")
	publish(client, "false, 13")		#Flawless Atas sobek Samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false14():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Wrinkled Atas dan Flawless Samping")
	publish(client, "false, 14") 		#Kerut atas Tanpa cacat samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false15():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Wrinkled Atas dan Wrinkled Samping")
	publish(client, "false, 15")		#Kerut atas Kerut Samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false16():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Wrinkled Atas dan Tear Samping")
	publish(client, "false, 16")		#Kerut atas Robek Samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false17():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Shifted Atas dan Flawless Samping")
	publish(client, "false, 17")		#Label tidak rapih atas Tanpa cacat samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false18():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Shifted Atas dan Wrinkled Samping")
	publish(client, "false, 18")		#Label tidak rapih atas Kerut Samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false19():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Shifted Atas dan Tear Samping")
	publish(client, "false, 19")		#Label tidak rapih atas Robek Samping
	unsubscribe(mqtt_client)
	x = []
	b =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false20():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Tear Atas dan Flawless Samping")
	publish(client, "false, 20")		#Robek atas Tanpa cacat Samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false21():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Tear Atas dan Wrinkled Samping")
	publish(client, "false, 21")		#Robek atas Kerut Samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def false22():
	global a,x,hitung,maksimal,paling,data
	print("Terdeteksi Tear Atas dan Tear Samping")
	publish(client, "false, 22")		#Robek atas Robek samping
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0
def salah():
	global a,x,hitung,maksimal,paling,data, C
	print("Tidak terdeteksi")
	unsubscribe(mqtt_client)
	x = []
	a =[]
	C = 0
	hitung = 0
	maksimal = 0
	paling = []
	data = []
	d=0

if __name__ == '__main__':
	client = connect_mqtt()
	client.loop_start()
	subscribe(client)
	subscribe1(client)
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
	net = cv2.dnn.readNet(model='top_best.weights', config='cfg/Tissue/top.cfg')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	if d == 0:
		print("Detecting...")
	d=d+1
	# Process image.
	while True:	
		subscribe(client)
		subscribe1(client)
		ret, frame = cap.read()
		if len(a) == 0:
			subscribe(client)
		if len(a) > 0:
			hitung = count_strings(a)
			maksimal=max(hitung.values())
			paling = [element for element, count in hitung.items() if count == maksimal]		#===================== Buat Jetson SIDE
			F= a.count('FlawlessSamping')
			W= a.count('WrinkledSamping')
			T= a.count('TearSamping')		
		
		if len(x) > 0:
			subscribe(client)
			C = data.count('true')
			counts = count_strings(x)
			max_count =max(counts.values())
			most_common = [element for element, count in counts.items() if count == max_count] #====================== Buat Jetson TOP
			Flawless= x.count('Flawless')
			Wrinkled= x.count('Wrinkled')
			Shifted = x.count('Shifted')
			Tear = x.count("Tear")
			if C >0:
				if a == []:
					salah()
				elif most_common == ['Flawless']:
					if Wrinkled > 0:
						subscribe1(client)
						print(paling)
						if paling == ['FlawlessSamping']:		#Kerut Atas Tanpa Cacat Samping
							if T > W:
								false16()
							elif T > 0:
								false16()
							elif W > 0:
								false15()
							else:
								false14()
						elif paling == ['WrinkledSamping']:		#Kerut Atas Kerut Samping
							false15()
						elif paling == ['TearSamping']:			#Kerut Atas Sobek Samping
							false16()
						elif F > 0:
							if F == W:
								false15()
							if F == T:
								false16()
							elif F == W == T:
								false16()
						elif W > 0:
							if W == T:
								false16()
							if W == F:
								false15()
						elif T > 0:
							if T == F:
								false16()
							if T == W:
								false16()
						elif a == []:
							salah()
					elif Shifted > 0:
						subscribe1(client)
						if paling == ['FlawlessSamping']:		#Label Tidak Rapi Atas Tanpa Cacat Samping
							if T > W:
								false19()
							elif T > 0:
								false19()
							elif W > 0:
								false18()
							else:
								false17()
						elif paling == ['WrinkledSamping']:		#Label Tidak Rapi Atas Kerut Samping
							false18()
						elif paling == ['TearSamping']:			#Label Tidak Rapi Atas Sobek Samping
							false19()
						elif F > 0:
							if F == W:
								false18()
							elif F == T:
								false19()
							elif F == W == T:
								false19()
						elif W > 0:
							if W == T:
								false19()
							elif W == F:
								false18()
						elif T > 0:
							if T == F:
								false19()
							elif T == W:
								false19()
						elif a == []:
							salah()
					elif Tear > 0:
						subscribe1(client)
						if paling == ['FlawlessSamping']:		#Robek Atas Tanpa Cacat Samping
							
							if T > W:
								false22()
							elif T > 0:
								false22()
							elif W > 0:
								false21()
							else:
								false20()
						elif paling == ['WrinkledSamping']:		#Robek Atas Kerut Samping
							false21()
						elif paling == ['TearSamping']:			#Robek Atas Robek Samping
							false22()
						elif F > 0:
							if F == W:
								false21()
							elif F == T:
								false22()
							elif F == W == T:
								false22()
						elif W > 0:
							if W == T:
								false22()
							elif W == F:
								false21()
						elif T > 0:
							if T == F:
								false22()
							elif T == W:
								false22()
						elif a == []:
							salah()
					else:
						subscribe1(client)
						if paling == ['FlawlessSamping']:		#Tanpa cacat atas Tanpa Cacat Samping
							if T > W:
								false13()
							elif T > 0:
								false13()
							elif W > 0:
								false12()
							else:
								true11()
						elif paling == ['WrinkledSamping']:		#Tanpa cacat atas kerut samping
							false12()
						elif paling == ['TearSamping']:			#Tanpa cacat atas sobek samping
							false13()
						elif F > 0:
							if F == W:
								false12()
							elif F == T:
								false13()
							elif F == W == T:
								false13()
						elif W > 0:
							if W == T:
								false13()
							if W == F:
								false12()
						elif T > 0:
							if T == F:
								false13()
							if T == W:
								false13()
						elif a == []:
							salah()
				elif most_common == ['Wrinkled']:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Kerut atas Tanpa cacat samping
						if T > W:
							false16()
						elif T > 0:
							false16()
						elif W > 0:
							false15()
						else:
							false14()
					elif paling == ['WrinkledSamping']:		#Kerut atas Kerut Samping
						false15()
					elif paling == ['TearSamping']:			#Kerut atas Robek Samping
						false16()
					elif F > 0:
						if F == W:
							false15()
						elif F == T:
							false16()
						elif F == W == T:
							false16()
					elif W > 0:
						if W == T:
							false16()
						elif W == F:
							false15()
					elif T > 0:
						if T == F:
							false16()
						elif T == W:
							false16()
					elif a == []:
						salah()
				elif most_common == ['Shifted']:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Label tidak rapih atas Tanpa cacat samping
						if T > W:
							false18()
						elif T > 0:
							false18()
						elif W > 0:
							false19()
						else:
							false17()
					elif paling == ['WrinkledSamping']:		#Label tidak rapih atas Kerut Samping
						false18()
					elif paling == ['TearSamping']:			#Label tidak rapih atas Robek Samping
						false19()
					elif F > 0:
						if F == W:
							false18()
						elif F == T:
							false19()
						elif F == W == T:
							false19()
					elif W > 0:
						if W == T:
							false19()
						elif W == F:
							false18()
					elif T > 0:
						if T == F:
							false19()
						elif T == W:
							false19()
					elif a == []:
						salah()
				elif most_common == ['Tear']:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Robek atas Tanpa cacat Samping
						if T > W:
							false22()
						elif T > 0:
							false22()
						elif W > 0:
							false21()
						else:
							false20()
					elif paling == ['WrinkledSamping']:		#Robek atas Kerut Samping
						false21()
					elif paling == ['TearSamping']:			#Robek atas Robek samping
						false22()
					elif F > 0:
						if F == W:
							false21()
						elif F == T:
							false22()
						elif F == W == T:
							false22()
					elif W > 0:
						if W == T:
							false22()
						elif W == F:
							false21()
					elif T > 0:
						if T == F:
							false22()
						elif T == W:
							false22()
					elif a == []:
						salah()
				elif Wrinkled == Shifted == Tear:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Robek atas Tanpa cacat Samping
						if T > W:
							false22()
						elif T > 0:
							false22()
						elif W > 0:
							false21()
						else:
							false20()
					elif paling == ['WrinkledSamping']:		#Robek atas Kerut Samping
						false21()
					elif paling == ['TearSamping']:			#Robek atas Robek samping
						false22()
					elif F > 0:
						if F == W:
							false21()
						elif F == T:
							false22()
						elif F == W == T:
							false22()
					elif W > 0:
						if W == T:
							false22()
						elif W == F:
							false21()
					elif T > 0:
						if T == F:
							false22()
						elif T == W:
							false22()
					elif a == []:
						salah()
				elif Wrinkled == Tear:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Robek atas Tanpa cacat Samping
						if T > W:
							false22()
						elif T > 0:
							false22()
						elif W > 0:
							false21()
						else:
							false20()
					elif paling == ['WrinkledSamping']:		#Robek atas Kerut Samping
						false21()
					elif paling == ['TearSamping']:			#Robek atas Robek samping
						false22()
					elif F > 0:
						if F == W:
							false21()
						elif F == T:
							false22()
						elif F == W == T:
							false22()
					elif W > 0:
						if W == T:
							false22()
						elif W == F:
							false21()
					elif T > 0:
						if T == F:
							false22()
						elif T == W:
							false22()
					elif a == []:
						salah()
				elif Wrinkled == Shifted:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Kerut atas Tanpa cacat samping
						if T > W:
							false16()
						elif T > 0:
							false16()
						elif W > 0:
							false15()
						else:
							false14()
					elif paling == ['WrinkledSamping']:		#Kerut atas Kerut Samping
						false15()
					elif paling == ['TearSamping']:			#Kerut atas Robek Samping
						false16()
					elif F > 0:
						if F == W:
							false15()
						elif F == T:
							false16()
						elif F == W == T:
							false16()
					elif W > 0:
						if W == T:
							false16()
						elif W == F:
							false15()
					elif T > 0:
						if T == F:
							false16()
						elif T == W:
							false16()
					elif a == []:
						salah()
				elif Shifted == Tear:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Robek atas Tanpa cacat Samping
						if T > W:
							false22()
						elif T > 0:
							false22()
						elif W > 0:
							false21()
						else:
							false20()
					elif paling == ['WrinkledSamping']:		#Robek atas Kerut Samping
						false21()
					elif paling == ['TearSamping']:			#Robek atas Robek samping
						false22()
					elif F > 0:
						if F == W:
							false21()
						elif F == T:
							false22()
						elif F == W == T:
							false22()
					elif W > 0:
						if W == T:
							false22()
						elif W == F:
							false21()
					elif T > 0:
						if T == F:
							false22()
						elif T == W:
							false22()
					elif a == []:
						salah()
				elif Flawless == Wrinkled == Shifted == Tear:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Robek atas Tanpa cacat Samping
						if T > W:
							false22()
						elif T > 0:
							false22()
						elif W > 0:
							false21()
						else:
							false20()
					elif paling == ['WrinkledSamping']:		#Robek atas Kerut Samping
						false21()
					elif paling == ['TearSamping']:			#Robek atas Robek samping
						false22()
					elif F > 0:
						if F == W:
							false21()
						elif F == T:
							false22()
						elif F == W == T:
							false22()
					elif W > 0:
						if W == T:
							false22()
						elif W == F:
							false21()
					elif T > 0:
						if T == F:
							false22()
						elif T == W:
							false22()
					elif a == []:
						salah()
				elif Flawless == Wrinkled == Shifted:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Kerut atas Tanpa cacat samping
						if T > W:
							false16()
						elif T > 0:
							false16()
						elif W > 0:
							false15()
						else:
							false14()
					elif paling == ['WrinkledSamping']:		#Kerut atas Kerut Samping
						false15()
					elif paling == ['TearSamping']:			#Kerut atas Robek Samping
						false16()
					elif F > 0:
						if F == W:
							false15()
						elif F == T:
							false16()
						elif F == W == T:
							false16()
					elif W > 0:
						if W == T:
							false16()
						elif W == F:
							false15()
					elif T > 0:
						if T == F:
							false16()
						elif T == W:
							false16()
					elif a == []:
						salah()
				elif Flawless == Wrinkled:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Kerut atas Tanpa cacat samping
						if T > W:
							false16()
						elif T > 0:
							false16()
						elif W > 0:
							false15()
						else:
							false14()
					elif paling == ['WrinkledSamping']:		#Kerut atas Kerut Samping
						false15()
					elif paling == ['TearSamping']:			#Kerut atas Robek Samping
						false16()
					elif F > 0:
						if F == W:
							false15()
						elif F == T:
							false16()
						elif F == W == T:
							false16()
					elif W > 0:
						if W == T:
							false16()
						elif W == F:
							false15()
					elif T > 0:
						if T == F:
							false16()
						elif T == W:
							false16()
					elif a == []:
						salah()
				elif Flawless == Shifted:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Kerut atas Tanpa cacat samping
						if T > W:
							false16()
						elif T > 0:
							false16()
						elif W > 0:
							false15()
						else:
							false14()
					elif paling == ['WrinkledSamping']:		#Kerut atas Kerut Samping
						false15()
					elif paling == ['TearSamping']:			#Kerut atas Robek Samping
						false16()
					elif F > 0:
						if F == W:
							false15()
						elif F == T:
							false16()
						elif F == W == T:
							false16()
					elif W > 0:
						if W == T:
							false16()
						elif W == F:
							false15()
					elif T > 0:
						if T == F:
							false16()
						elif T == W:
							false16()
					elif a == []:
						salah()
				elif Flawless == Tear:
					subscribe1(client)
					if paling == ['FlawlessSamping']:		#Robek atas Tanpa cacat Samping
						if T > W:
							false22()
						elif T > 0:
							false22()
						elif W > 0:
							false21()
						else:
							false20()
					elif paling == ['WrinkledSamping']:		#Robek atas Kerut Samping
						false21()
					elif paling == ['TearSamping']:			#Robek atas Robek samping
						false22()
					elif F > 0:
						if F == W:
							false21()
						elif F == T:
							false22()
						elif F == W == T:
							false22()
					elif W > 0:
						if W == T:
							false22()
						elif W == F:
							false21()
					elif T > 0:
						if T == F:
							false22()
						elif T == W:
							false22()
					elif a == []:
						salah()
				elif len(a) == 0:
					salah()
				elif a == [] :
					if b == 'true':
						salah()
						x = []
				else:
					salah()
			elif a == [] :
				if b == 'true':
					salah()
					x = []
				elif C > 0:
					salah()
		elif data == 'true':
			salah()
		detections = pre_process(frame, net)
		img = post_process(frame.copy(), detections)
		# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
		cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, GREEN, THICKNESS, cv2.LINE_AA)
		cv2.imshow('Output', img)	
		if cv2.waitKey(1) & 0xFF == ord('q'):	
			break
	cap.release()
	cv2.destroyAllWindows()
