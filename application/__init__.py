from flask import Flask
from flask_mysqldb import MySQL
import pickle
import cv2
import face_recognition
import random
import json
import torch
from datetime import datetime
# from config import Config
# from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate

from module_chat import NeuralNet
from preprocessing_chat import bag_of_words, tokenize

app = Flask(__name__)
# app.config.from_object(Config)
# db = SQLAlchemy(app)
# migrate = Migrate(app, db)

# Database
mysql = MySQL(app)
app.secret_key = 'super secret key'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'myabsensi'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# chatbot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
train = torch.load(FILE)

input_size = train["input_size"]
hidden_size = train["hidden_size"]
output_size = train["output_size"]
all_words = train['all_words']
tags = train['tags']
model_state = train["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

        return "Maaf, kami tidak mengerti pertanyaan anda"

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)


# face_recognition
data = pickle.loads(open('training.dat', "rb").read())

#realtime absensi
def markAttendance(name):
    with open('absensi.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            # now= datetime.datetime.now()
            # today=now.day
            # month=now.month
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

name = data["names"]

# cap = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


class Recognition(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    def gen_frames(self):
        while True:
            # rval = False
            # while(not rval):
            #     (rval,frame) = cap.read()
            #     if(not rval):
            #         print("Gagal untuk membuka kamera. Coba lagi...")
            # startTime = time.time()
            success, frame = self.cap.read()  # read the camera frame
            # cv2.imshow('Video', frame)
            # camera.release()
            # cv2.destroyAllWindows()

            if not success:
                return None
                # break
            else:
                # if success:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                # R, G, B = small_frame[:,:,0], small_frame[:,:,1], small_frame[:,:,2]
                # imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(
                        data["encodings"], face_encoding)
                    name = "Tidak Diketahui"

                    if True in matches:
                        # Find positions at which we get True and store them
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        # loop over the matched indexes and maintain a count for
                        # each recognized face face
                        for i in matchedIdxs:
                            # Check the names at respective indexes we stored in matchedIdxs
                            name = data["names"][i]
                            # increase count for the name we got
                            counts[name] = counts.get(name, 0) + 1
                            # set name which has highest count
                        name = max(counts, key=counts.get)

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(
                        data["encodings"], face_encoding)
                    print(matches, face_distances)

                    # best_match_index = np.argmin(face_distances)
                    # if matches[best_match_index]:
                    #     name = known_face_names[best_match_index]

                    face_names.append(name)

                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom-35),
                                  (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left+6, bottom-6),
                                font, 1, (255, 255, 255), 2)
                    # Akurasi
                    cv2.putText(
                        frame, f' {round(face_distances[0],2)}', (50, 50), font, 1, (0, 0, 255), 2)
                    markAttendance(name)

                ret, buffer = cv2.imencode('.jpg', frame)
                return buffer.tobytes()

from application import routes
