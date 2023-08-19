from flask import Flask, request, jsonify
import os
import json
import requests
from PIL import Image
import cv2
import torch
import numpy as np
from io import BytesIO
import torchvision.transforms as T
from facenet_pytorch import MTCNN, InceptionResnetV1
from model import siamese_model  # Import your siamese_model class or function

app = Flask(__name__)

# Set paths and configurations
siamese_model_path = "saved_models/siamese_model"
db_path = "database/"
database_embeddings_path = os.path.join(db_path, "database_embeddings")
device = "cuda" if torch.cuda.is_available() else "cpu"
margin = 0
THRESHOLD = 0.4
file_path = "classes.txt"

# Load the models and define required functions
mtcnn = MTCNN(image_size=128, margin=margin).eval()
resnet = InceptionResnetV1(pretrained="vggface2").to(device).eval()
loader = T.Compose([T.ToTensor()])

# Load Siamese Model
model = siamese_model()
model.load_state_dict(torch.load(siamese_model_path, map_location=torch.device('cpu')))
model.eval()
model.to(device)

def process_input_image(input_img, mtcnn, resnet):
    boxes, probs, points = mtcnn.detect(input_img, landmarks=True)
    print(boxes)
    if boxes is not None and len(boxes) > 0:
        bbox = boxes[0] 
        input_img = np.array(input_img)
        box = (np.array(bbox)).astype(int)
        print("bbox",bbox)
        cropped_face = input_img[box[1] : box[3] + 1, box[0] : box[2] + 1]
        input_img = cv2.resize(cropped_face, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        input_img = loader((input_img - 127.5) / 128.0).type(torch.FloatTensor)  # Normalizing and converting to tensor
        face_embedding = resnet(input_img.unsqueeze(0).to(device)).reshape((1, 1, 512))
        return face_embedding,bbox
    return None,None

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Face Recognition API!"
# Define a route for the API
@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        print("api call")
        classes=[]
        uploaded_image = request.files['image']
        flag = int(request.form['flag'])
        print(flag)

        filename="image.jpg"
        # Open the image using PIL
        image = Image.open(uploaded_image)
        image_full_path = os.path.join('uploads', filename)
        image.save(image_full_path)

        flag=0
        input_img = Image.open(image_full_path)
        face_embedding,image_bbox = process_input_image(input_img, mtcnn, resnet)

        if flag == 1:
            # Remove existing embedding file and recalculate embeddings
            if os.path.exists(database_embeddings_path):
                os.remove(database_embeddings_path)
            
            # Calculate and save new embeddings for all images in the database
            reference_cropped_img = []
            print("classes",classes)
            for i in os.listdir(db_path):
                classes.append(i)

            for i in classes:
                reference_img = Image.open(os.path.join(db_path, i, os.listdir(os.path.join(db_path, i))[0]))
                boxes, probs, points = mtcnn.detect(reference_img, landmarks=True)

                boxes = (np.array(boxes[0])).astype(int)
                input_img = np.array(reference_img)[
                    boxes[1] : boxes[3] + 1, boxes[0] : boxes[2] + 1
                ].copy()
                input_img = cv2.resize(
                    input_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC
                )
                input_img = loader((input_img - 127.5) / 128.0).type(torch.FloatTensor)
                reference_cropped_img.append(input_img)

            try:
                torch.save({"reference": reference_cropped_img}, database_embeddings_path)
                # Open the file in write mode
                with open(file_path, "w") as file:
                    # Iterate through the list and write each element to a new line
                    for item in classes:
                        file.write(item + "\n")
                print("Embeddings saved successfully !!!")
            except:
                print("Already exist")
        else:
            # Open the file in write mode
            with open(file_path, "r") as file:
                 # Read each line and append it to the list
                classes = [line.strip() for line in file]
        if face_embedding is not None:
            # Perform face recognition and similarity check
            reference_embeddings = torch.load(database_embeddings_path)["reference"]
            similarity_list = []
            # i=0
            for ref_embedding in reference_embeddings:
                ref_embedding = resnet(ref_embedding.unsqueeze(0).to(device)).reshape((1, 1, 512))
                similarity = model(ref_embedding.to(device), face_embedding.to(device)).item()
                similarity_list.append(similarity)
            max_similarity = max(similarity_list)
            label = classes[similarity_list.index(max_similarity)]
            image_bbox=list(image_bbox)
            if max_similarity >= THRESHOLD:
                result = {"status": "success", "message": "Face recognized", "label": label,"similarity_value":max_similarity,"bbox":image_bbox}
            else:
                result = {"status": "success", "message": "Face not recognized", "label": "unknown","similarity_value":max_similarity,"bbox":image_bbox}
        else:
            result = {"status": "error", "message": "No face detected in the uploaded image", "label": "unknown","similarity_value":0,"bbox":None}
        
        if image_full_path is not None and os.path.exists(image_full_path):
            os.remove(image_full_path)
            print("Cropped image deleted:", image_full_path)
            
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "label": "unknown","similarity_value":0,"bbox":None})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(host='0.0.0.0', port=5000,debug=True)
