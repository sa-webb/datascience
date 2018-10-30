import socket

HOST = 'localhost'
PORT = 9876
ADDR = (HOST,PORT)
BUFSIZE = 4096

# Path to mp4 file 
videofile = "videos/royalty-free_footage_wien_18_640x360.mp4"

bytes = open(videofile).read()

print(len(bytes))

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)
client.send(bytes)
client.close()