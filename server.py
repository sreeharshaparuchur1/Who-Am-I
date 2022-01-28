import numpy as np
import cv2
import socket
import pickle
import struct
from Recognize import *
import threading
import ClientHandler
import tqdm
import shutil
import sys
import os
import time
import matplotlib.pyplot as plt
from Train import *

HOST = "localhost"
# Port for socket
PORT = 5015 # Arbitrary non-privileged port

def send_one_message(sock, data):
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)

def quit(command):
    sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sckt.connect((HOST, PORT))
    send_one_message(sckt, command.encode('utf-8'))
    sckt.close()
    return

def listen(server_socket):
    while True:
        print('\r[CONN] Waiting for client...')
        client_socket, addr = server_socket.accept()
        print('\r[CONN] Connected from ip:',addr[0], 'and port : ',addr[1])
        t = threading.Thread(target=ClientHandler.handle_client, args=(client_socket,))
        #starting new thread
        t.start()
        if not t.is_alive(): 
            print('[THREAD] Serviced Thread.')
            break

def train(video):
	
    face_cascade = './Haar_Cascades/haarcascade_frontalface_default.xml'
    if not (os.path.isfile(face_cascade)):
        raise RuntimeError("%s: not found" % face_cascade)

    right_eye_cascade = './Haar_Cascades/haarcascade_righteye_2splits.xml'
    if not (os.path.isfile(right_eye_cascade)):
        raise RuntimeError("%s: not found" % right_eye_cascade)

    left_eye_cascade = './Haar_Cascades/haarcascade_lefteye_2splits.xml'
    if not (os.path.isfile(left_eye_cascade)):
        raise RuntimeError("%s: not found" % left_eye_cascade)

    var = [1,8,8,8]#radius,neighbour,gx,gy
    model = Train_Model(face_cascade,right_eye_cascade,left_eye_cascade,var)
    model.create_dataset(50,video,'dataset/')
    model.train('dataset/','train.yaml')

def TakeInput(name, server_socket):
    while True:
        sys.stdout.write('%s@[Server] -> ' %name)
        sys.stdout.flush()
        command = sys.stdin.readline().strip()

        if (command == 'quit'):
            print('[WARNING] Quitting Server side. If you sent this command in between an operation you might experience bugs. You have been warned.')
            #quit()
            break

        elif (command == 'trainVideo'):
            video_name = input('Enter the location to your video : ')
            # dataset_name = input("What would you like to call this person? : ")
            video = cv2.VideoCapture(video_name)
            train(video)

        elif (command == 'listen'):
            print("\r[LISN] Socket is now listening")
            listen(server_socket)

        elif (command == 'trainWebc'):
            print('[WARNING] We hope you have a webcam and it is detected by your machine. Running at 640 x 480.')
            video = cv2.VideoCapture(0)
            video.set(3, 640)
            video.set(4, 480)
            # dataset_name = input("What would you like me to call you as? : ")
            train(video)

def main():
    try:
        # Create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("\r[CONN] Socket successfully created")
    except socket.error as err:
        print("\r[FAIL] Socket creation failed with error : ",err)

    try:
        server_socket.bind((HOST, PORT))
    except socket.error as err:
        print('[FAIL] Bind failed. Error Message :  ',err)
        sys.exit()

    print('Socket bind successfully')
    print("\r[BIND] Socket binded to : ",PORT)

    # Listen for connections : allow only 5 connection
    server_socket.listen(5)
    print("\r[RDY] Socket is now deployed") 	 

    name = input('Enter your name :')
    print('Welcome to WhoAmI. This is server.')
    TakeInput(name, server_socket)
    
if __name__=="__main__":
    main()