import os
import time
import psutil
from werkzeug.utils import secure_filename
# using datetime module
import datetime;
import numpy as np
import cv2
# from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify,render_template,redirect
import requests
from flask_cors import CORS,cross_origin
import tensorflow as tf
from keras.models import load_model
# from scipy.spatial import distance
# from keras.utils import image
from keras.preprocessing import image
from sklearn.metrics.pairwise import euclidean_distances
from pydub import AudioSegment
import tempfile
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
import numpy
import moviepy.editor
import requests
import firebase_admin
from firebase_admin import credentials, firestore, storage, initialize_app
# model = load_model('vgg16SelamPayVe/rsion.h5')
model = load_model('vgg16SelamPayCompiledVersion.h5')
# import pyrebase

facedefault_path_cascade = "detectors/haarcascade_frontalface_default.xml"
facedefault_cascade = cv2.CascadeClassifier(facedefault_path_cascade)


face_path_cascade = "detectors/haarcascade_frontalface_alt2.xml"
facedefault_path_cascade = "detectors/haarcascade_frontalface_default.xml"
eyes_path_cascade = "detectors/haarcascade_eye.xml"

# defining face detector
face_cascade = cv2.CascadeClassifier(face_path_cascade)
facedefault_cascade = cv2.CascadeClassifier(facedefault_path_cascade)
# defining eye detector
eyes_cascade = cv2.CascadeClassifier(eyes_path_cascade)

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

# cred = credentials.Certificate("credencial/pointagepfe-firebase-adminsdk-xzbw2-93d3b85438.json")
cred = credentials.Certificate("credencial/pfepointage.json")

# firebase_admin.initialize_app(cred, {
#     'storageBucket': 'pointagepfe.appspot.com'
# })
firebase_admin.initialize_app(cred, {
    'storageBucket': 'pfepointage.appspot.com'
})

db = firestore.client()

#fire bases collections
attendances_ref = db.collection('attendances')
employees_ref = db.collection('employees')
# Get a reference to the bucket
bucket = storage.bucket()





app = Flask(__name__)
cors = CORS(app)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

basedir = os.path.abspath(os.path.dirname(__file__))

# Create a directory in a known location to save files to.
# uploads_dir = os.path.join(app.instance_path, 'uploads')
# os.makedirs(uploads_dir, exist_ok=True)


@app.route('/enregistrer' , methods=['POST'])
@cross_origin(origin="*")
@cross_origin(methods=['GET', 'POST'])
def enregistrer():

    file = request.files['file']
    nom = request.form['nom']
    telephone = request.form['telephone']
    nni = request.form['nni']
    matricule = request.form['matricule']
    role = request.form['role']

    
    id = datetime.datetime.now().timestamp()

    imagename = f"{id}_{nom}.jpg"
    audioname = f"{id}_{nom}.wav"

    a,b = str(id).split(".")
    input_videoName = a + b +".mp4"
    print(f"input_videoName : {input_videoName}")

    
    # file.save(os.path.join(uploads_dir, secure_filename("input_video.mp4")))
    file.save(os.path.join("instance/uploads/", secure_filename(input_videoName)))
    video_path = os.path.join("instance/uploads/", secure_filename(input_videoName))

    try:
        image_local_path = extractBestImage(video_path)
        image_sorage_path = os.path.join('employeesImages/', imagename)
        image_url = uploadFile(image_sorage_path,image_local_path)
        audio_local_path = extractAudio(video_path)
        audio_sorage_path = os.path.join('employeesAudios/', audioname)
        audio_url = uploadFile(audio_sorage_path,audio_local_path)

        os.remove(f"instance/uploads/{input_videoName}")
        employee = {
            "id":id,
            "nom" : nom,
            "telephone" : telephone,
            "nni":nni,
            "matricule": matricule,
            "role": role,
            "image_url":image_url,
            "audio_url":audio_url
        }

        employees_ref.document(str(id)).set(employee)

        return jsonify({"success": True}), 200
    except Exception as e:
        print("KU")
        os.remove(f"instance/uploads/{input_videoName}")
        return jsonify({f"Exception, {e}": False}), 500



@app.route('/getEmployees', methods=['GET'])
@cross_origin(origin="*")
def getEmployees():
    """
        read() : Fetches documents from Firestore collection as JSON
        todo : Return document that matches query ID
        all_todos : Return all documents
    """
    try:
        # Check if ID was passed to URL query
        employee_id = request.args.get('id')    
        if employee_id:
            employee = employees_ref.document(employee_id).get()
            return jsonify(employee.to_dict()), 200
        else:
            all_employees = [doc.to_dict() for doc in employees_ref.stream()]
            return jsonify(all_employees), 200
    except Exception as e:
        return f"An Error Occured: {e}"

@app.route('/updateEmployee', methods=['POST', 'PUT'])
@cross_origin(origin="*")
def updateEmployee():
    """
        update() : Update document in Firestore collection with request body
        Ensure you pass a custom ID as part of json body in post request
        e.g. json={'id': '1', 'title': 'Write a blog post today'}
    """
    try:
        id = request.json['id']
        employees_ref.document(str(id)).update(request.json)
        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route('/deleteEmployee', methods=['GET', 'DELETE'])
@cross_origin(origin="*")
def deleteEmployee():
    """
        delete() : Delete a document from Firestore collection
    """
    try:
        # Check for ID in URL query
        employee_id = request.args.get('id')
        employees_ref.document(str(employee_id)).delete()
        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route('/pointerEmployee' , methods=['POST'])
@cross_origin(origin="*")
@cross_origin(methods=['GET', 'POST'])
def pointagerEmployee():
    file = request.files['file']
    id = datetime.datetime.now().timestamp()
    n,m = str(id).split(".")
    input_videoName = n + m +".mp4"
    print(f"input_videoName : {input_videoName}")
    # file.save(os.path.join(uploads_dir, secure_filename(input_videoName)))
    file.save(os.path.join("instance/uploads/", secure_filename(input_videoName)))
    timesTemp = str(datetime.datetime.now().timestamp())
    a,b = timesTemp.split(".")
    inputName = a + b
    print(f"imageName : {inputName}")
    # filename = f"{id}_{nom}"
    imagename = f"{inputName}.jpg"
    audioname = f"{inputName}.wav"
    # video_path = os.path.join(uploads_dir, secure_filename(input_videoName))
    video_path = os.path.join("instance/uploads/", secure_filename(input_videoName))
    pointedEmployee = {}

    try:
        image_local_path = extractBestImage(video_path)
        image_sorage_path = os.path.join('inputImages/', imagename)
        input_image_url = uploadFile(image_sorage_path,image_local_path)
        audio_local_path = extractAudio(video_path)
        audio_sorage_path = os.path.join('inputAudios/', audioname)
        input_audio_url = uploadFile(audio_sorage_path,audio_local_path)

        # clean instance/upload
        os.remove(f"instance/uploads/{input_videoName}")
        all_employees = [doc.to_dict() for doc in employees_ref.stream()]
        for employee in all_employees:
            image_prediction = pointagerParImage(input_image_url, employee["image_url"])
            print(f"image_prediction{image_prediction}")
            audio_prediction = pointageParAudio(input_audio_url, employee["audio_url"])
            print(f"audio_prediction{audio_prediction}")
            
            # Clean up the automatically stored file after verification
            # chain.replace(" ","%20") pour remplacer les espaces(" ") par ("%20")
            userNameFromAudio = employee['nom'].replace(" ", "%20")

            verificatio_audioname = f"{employee['id']}_{userNameFromAudio}.wav"
            os.remove(verificatio_audioname)
            os.remove(audioname)


            if image_prediction == True and audio_prediction == True :
                pointedEmployee = employee
                break
            elif image_prediction == True or audio_prediction == True :
                pointedEmployee = employee
                break
        if pointedEmployee != {}:
            attendance = {
                "id": id,
                "nom_employee" : pointedEmployee["nom"],
                "id_employee" : pointedEmployee["id"],
                "clockIn" : id,
                "phoneUser" : pointedEmployee["telephone"],
                "role" : pointedEmployee["role"],
                # "employee": pointedEmployee
            }
            attendances_ref.document(str(id)).set(attendance)
            # Convert the timestamp to a `datetime` object
            time = datetime.datetime.fromtimestamp(attendance['clockIn'])
            attendance['clockIn'] = time
            # deleteFileFromStorage(input_image_url)
            # deleteFileFromStorage(input_audio_url)
            return jsonify(attendance), 200
        else:
            return jsonify({"Erreur, il n'est pas pointer": True}), 200
    except Exception as e:
        print("KU")
        os.remove(f"instance/uploads/{input_videoName}")
        return jsonify({f"Exception, {e}": False}), 500

@app.route('/depointerEmployee' , methods=['POST'])
@cross_origin(origin="*")
@cross_origin(methods=['GET', 'POST'])
def depointerEmployee():

    
    file = request.files['file']
    timpsTemp = datetime.datetime.now().timestamp()
    n,m = str(timpsTemp).split(".")
    inputName = n + m
    input_videoName = inputName +".mp4"
    print(f"input_videoName : {input_videoName}")
    # file.save(os.path.join(uploads_dir, secure_filename(input_videoName)))
    file.save(os.path.join("instance/uploads/", secure_filename(input_videoName)))
    imagename = f"{inputName}.jpg"
    audioname = f"{inputName}.wav"
    # video_path = os.path.join(uploads_dir, secure_filename(input_videoName))
    video_path = os.path.join("instance/uploads/", secure_filename(input_videoName))

    try:
        image_local_path = extractBestImage(video_path)
        image_sorage_path = os.path.join('inputImages/', imagename)
        input_image_url = uploadFile(image_sorage_path,image_local_path)
        audio_local_path = extractAudio(video_path)
        audio_sorage_path = os.path.join('inputAudios/', audioname)
        input_audio_url = uploadFile(audio_sorage_path,audio_local_path)

        os.remove(f"instance/uploads/{input_videoName}")
        pointedEmployee = {}
        all_employees = [doc.to_dict() for doc in employees_ref.stream()]

        for employee in all_employees:
            image_prediction = pointagerParImage(input_image_url, employee["image_url"])
            audio_prediction = pointageParAudio(input_audio_url, employee["audio_url"])
            if image_prediction == True or audio_prediction == True :
                pointedEmployee = employee
                break
        if pointedEmployee != {}:
        
            attendance = getAttendanceByUserId(pointedEmployee["id"])
            
            clockOut = datetime.datetime.now().timestamp()
            duration = clockOut - attendance['clockIn']
            print(f"attendance : {attendance}")
            print(f"attendance['id'] : {attendance['id']}")
            depointe_attendance = {
                "id": attendance["id"],
                "clockOut" : clockOut,
                "duration" : duration
            }
            attendances_ref.document(str(attendance["id"])).update(depointe_attendance)
            resultat_depointage = attendances_ref.document(str(attendance["id"])).get().to_dict()
            # Convert the timestamp to a `datetime` object
            start_time = datetime.datetime.fromtimestamp(resultat_depointage['clockIn'])
            resultat_depointage['clockIn'] = start_time
            end_time = datetime.datetime.fromtimestamp(resultat_depointage['clockOut'])
            resultat_depointage['clockOut'] = end_time

            # deleteFileFromStorage(input_image_url)
            # deleteFileFromStorage(input_audio_url)

            # # # Calculate the duration between the two dates
            # duration_notTimsTemp = end_time - start_time
            # resultat_depointage['duration'] = duration_notTimsTemp
            # print("Duration: {} days, {} hours, {} minutes, {} seconds".format(duration_notTimsTemp.days, duration_notTimsTemp.seconds//3600, (duration_notTimsTemp.seconds//60)%60, duration_notTimsTemp.seconds%60))
            return jsonify(resultat_depointage), 200
        else:
            return jsonify({"Erreur, il n'est pas pointer": True}), 200
    except Exception as e:
        print("KU")
        os.remove(f"instance/uploads/{input_videoName}")
        return jsonify({f"Exception, {e}": False}), 500
    

def getAttendanceByUserId(id_employee):
    try:
        query = attendances_ref.where(u'id_employee', u'==', id_employee)
        results = query.stream()
        all_attendances = [doc.to_dict() for doc in results]
        lastAttendance = getLastAttendance(all_attendances)
        return lastAttendance
    except Exception as e:
        return f"An Error Occured: {e}"

def getLastAttendance(listAttendances):
    max = 0
    lastAttendance = {}
    for attendance in listAttendances:
        if max < attendance['id']:
            max = attendance['id']
            lastAttendance = attendance
    return lastAttendance


#################################################### Services ##################################################

def pointagerParImage(input_image_url, validation_image_url):
    # Make Predictions 
    known_embedding = getEmbedding(input_image_url,model)
    print(f"known_embedding {known_embedding}")
    candidate_embedding= getEmbedding(validation_image_url,model)
    print(f"candidate_embedding {candidate_embedding}")
    result = compareEmbeddings(known_embedding, candidate_embedding, 0.5)
    return result

def pointageParAudio(input_audio_url, validation_audio_url):
    # predire the speaker and get score and prediction
    score, prediction = verification.verify_files(input_audio_url, validation_audio_url)
    return prediction[0].item()

def getEmbedding(img_path, model):

    # Download the image from the URL
    response = requests.get(img_path)

    timesTemp = str(datetime.datetime.now().timestamp())
    a,b = timesTemp.split(".")
    imageName = a + b
    print(f"imageName : {imageName}")

    # Save the image to disk
    with open(f'temporaryStorage/{imageName}.jpg', 'wb') as f:
        f.write(response.content)


    # img = load_img(image_url)
    img = tf.keras.utils.load_img(f'temporaryStorage/{imageName}.jpg', target_size=(100, 100))

    # Clean up the temporary file
    os.remove(f'temporaryStorage/{imageName}.jpg')

    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)


    embedding = model.predict(x)
 
    print(f"the vector embeddinh is :{embedding}")
    return embedding

# determine if a candidate face is a match for a known face
def compareEmbeddings(known_embedding, candidate_embedding, thresh):
    
    # calculate Euclidean distance between embeddings
    score = euclidean_distances(known_embedding, candidate_embedding)
    # print score result
    print(f"score is {score}")
    if score <= thresh:
        
            verified=True
            return verified
    else:
            verified=False
            return verified


#################################extract images and audios from Videos##############################################3
#extraire l'image la plus significative par analyse de mouvement
def extractBestImage2(video_path):

    # Charger une vidéo
    cap = cv2.VideoCapture(video_path)

    # Initialiser l'analyseur de mouvement
    motion_detector = cv2.createBackgroundSubtractorMOG2()

    # Initialiser les variables de mouvement
    max_motion = 0
    max_motion_frame = None

    # Boucle sur chaque frame de la vidéo
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Détection de mouvement sur la frame actuelle
        mask = motion_detector.apply(frame)

        # Calculer la somme des pixels blancs dans le masque
        motion = cv2.countNonZero(mask)

        # Si le mouvement est plus grand que le précédent, sauvegarder la frame
        if motion > max_motion:
            max_motion = motion
            max_motion_frame = frame


    timpsTemp = str(datetime.datetime.now().timestamp())
    n,m = timpsTemp.split(".")
    significant_image_name = n + m
    cv2.imwrite(f'extractedData/significantImageExtracted/{significant_image_name}.jpg', max_motion_frame)
    print("image stored")
    extracted_image_path = f'extractedData/significantImageExtracted/{significant_image_name}.jpg'
    return extracted_image_path



def extractAudio (video_path):
    timpsTemp = str(datetime.datetime.now().timestamp())
    n,m = timpsTemp.split(".")
    audio_name = n + m
    video = moviepy.editor.VideoFileClip(video_path)
    audio = video.audio
    # Fermer la vidéo
    # video.close()
    # os.close(video_path)
    audio.write_audiofile(f'extractedData/audio/{audio_name}.mp3')
    extracted_audio_path = f'extractedData/audio/{audio_name}.mp3'
    converted_audio_path = f"extractedData/converted_audio/{audio_name}.wav"
    # convert mp3 to wav                                                            
    sound = AudioSegment.from_mp3(extracted_audio_path)
    sound.export(converted_audio_path, format="wav")
    video.close()
    print("closed video")
    print(converted_audio_path)
    os.remove(f'extractedData/audio/{audio_name}.mp3')
    return converted_audio_path

##################extraire l'image la plus significative par simple algorithm######################################################
def extractBestImage1(video_path):
    cap=cv2.VideoCapture(video_path)
    count=0

    # Initialiser la liste des images significatives
    significant_images = []

    while cap.isOpened():
        t,frame = cap.read()
        frame_without_rect = frame
        if frame is not None:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            facedefault = facedefault_cascade.detectMultiScale(gray,1.3,5)
            eyes = eyes_cascade.detectMultiScale(gray,1.3,5)

            if len(facedefault) == 0:
                continue

            if len(facedefault) > 0:
                cv2.imwrite("extractedData/images/a"+str(count)+".jpg",frame)
                # Trouver le visage le plus grand
                max_face = max(facedefault, key=lambda x: x[2] * x[3])
                # Charger une image de référence
                image_notrect = cv2.imread("extractedData/images/a"+str(count)+".jpg")

                # Extraire l'image du visage
                x, y, w, h = max_face
                face_image = image_notrect[y-65:y+h+65, x+15:x+w-15]

                # Ajouter l'image à la liste des images significatives
                significant_images.append(face_image)
            
            
            

        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        
        count=count+1
        # cv2.waitKey(30)

    # Calculer le nombre de pixels dans chaque image significative
    pixel_counts = [face_image.shape[0] * face_image.shape[1] for face_image in significant_images]

    # Sélectionner l'image la plus significative comme celle avec le plus grand nombre de pixels
    significant_index = pixel_counts.index(max(pixel_counts))
    significant_image = significant_images[significant_index]

    # Enregistrer l'image significative
    cv2.imwrite('extractedData/significant/significant_image.jpg', significant_image)
    extracted_image_path = 'extractedData/significant/significant_image.jpg'
    return extracted_image_path


###########################


def extractBestImage(video_path):
    cap=cv2.VideoCapture(video_path)
    count=0

    # Initialiser la liste des images significatives
    significant_images = []

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total number of frames in video
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # Get frame rate
    max_iterations = video_length // fps # Calculate maximum number of iterations based on video length and frame rate


    while cap.isOpened():
        t,frame = cap.read()
        frame_without_rect = frame
        if frame is not None:
            # frame = cv2.resize(frame,(600,400))
            # Rotate the image by 180 degrees
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            # Increase Image Resolution
            frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Improve Lighting
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

            # Increase Contrast
            gray = cv2.equalizeHist(gray)
            # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray,1.3,5)
            facedefault = facedefault_cascade.detectMultiScale(gray,1.3,5)
            eyes = eyes_cascade.detectMultiScale(gray,1.3,5)

            if len(facedefault) == 0:
                continue

            if len(facedefault) > 0 and len(face) > 0:
                temp = str(datetime.datetime.now().timestamp())
                a,b = temp.split(".")
                img_name = a + b
                cv2.imwrite("extractedData/images/a"+img_name+str(count)+".jpg",frame)
                # Trouver le visage le plus grand
                max_face = max(facedefault, key=lambda x: x[2] * x[3])
                # Charger une image de référence
                image_notrect = cv2.imread("extractedData/images/a"+img_name+str(count)+".jpg")

                
                # Extraire l'image du visage
                # x, y, w, h = max_face
                # face_image = image_notrect[y:y+h, x:x+w]

                # Ajouter l'image à la liste des images significatives
                significant_images.append(image_notrect)
                os.remove("extractedData/images/a"+img_name+str(count)+".jpg")
                print("removed image")
        
        
        count=count+1
        print(f"count : {count}")
        # cv2.waitKey(30)
        if count == max_iterations: # Condition to break out of the loop based on video length
            break
            
        cv2.waitKey(30)

    # Calculer le nombre de pixels dans chaque image significative
    pixel_counts = [face_image.shape[0] * face_image.shape[1] for face_image in significant_images]

    # Sélectionner l'image la plus significative comme celle avec le plus grand nombre de pixels
    significant_index = pixel_counts.index(max(pixel_counts))
    significant_image = significant_images[significant_index]

    # Enregistrer l'image significative
    timpsTemp = str(datetime.datetime.now().timestamp())
    n,m = timpsTemp.split(".")
    significant_image_name = n + m
    cv2.imwrite(f'extractedData/significantImageExtracted/{significant_image_name}.jpg', significant_image)
    print("image stored")
    extracted_image_path = f'extractedData/significantImageExtracted/{significant_image_name}.jpg'
    return extracted_image_path

#####################################upload files####################################################
def deleteFileFromStorage(file_url):
    try:
        blob = bucket.blob(file_url)

        blob.delete()
        return True
    except Exception as e:
        print(e)
        return False
    


def uploadFile(filename, filepath):


    blob = bucket.blob(filename)
    blob.upload_from_filename(filepath)

    # Opt : if you want to make public access from the URL
    blob.make_public()
    url = blob.public_url
    print("your file url", blob.public_url)
    os.remove(filepath)
    return url

def uploadInputData(filename, filepath):


    blob = bucket.blob(filename)
    blob.upload_from_filename(filepath)

    # Opt : if you want to make public access from the URL
    blob.make_public()
    url = blob.public_url
    print("your file url", blob.public_url)

    return url


###########run app ###########

if __name__ == '__main__':
    
    # app.run(debug=True,)
    app.run(host='0.0.0.0', port=5000,debug=True,threaded = True)