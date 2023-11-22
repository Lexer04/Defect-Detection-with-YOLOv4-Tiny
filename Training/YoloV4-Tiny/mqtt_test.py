import paho.mqtt.client as mqtt

# Define callback functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        # Subscribe to topics
        client.subscribe("detection/jetson") #Antar Jetson
        client.subscribe("ps_detect") #Photosensor1
        client.subscribe("status_product") #Kirim PLC
        client.subscribe("speed") #Kecepatan Konveyor
    else:
        print("Connection failed with error code " + str(rc))

def on_message(client, userdata, message):
    print("Received message on topic " + message.topic + ": " + str(message.payload))

# Create a new MQTT client instance
client = mqtt.Client()

# Set the username and password
client.username_pw_set("sorting", "fiHyn{odOmlap3@sorting")

# Set the callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the broker
client.connect("146.190.106.65", 1883)

# Start the network loop
client.loop_forever()
