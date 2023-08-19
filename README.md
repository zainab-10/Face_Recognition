# Face_Recognition
Face Recognition using MTCNN, InceptionResnetV1, and Siamese model.

## Face Recognition API
This project implements a Face Recognition API using the Flask web framework, allowing users to perform face recognition on uploaded images and determine if the face belongs to a known individual or is unknown. The project leverages advanced deep-learning models for both face detection and recognition.

### Face Detection
The **MTCNN (Multi-task Cascaded Convolutional Networks)** model is employed for face detection. MTCNN is a widely-used deep learning model that detects faces and facial landmarks in images. It can detect multiple faces within an image while providing bounding box coordinates and facial landmarks.

### Face Recognition
For face recognition, the project employs the **InceptionResnetV1** model, pre-trained on the VGGFace2 dataset. InceptionResnetV1 is a powerful deep neural network architecture designed to extract facial features in a way that captures intricate details, making it suitable for accurate face recognition. The model produces high-dimensional embeddings that encode the unique characteristics of each individual's face.

Additionally, the project uses a **Siamese** model for face similarity calculation. A Siamese network takes in pairs of images and determines whether they are of the same person or not. This architecture is ideal for face verification tasks, as it can learn to differentiate between similar and dissimilar faces.

## Future Improvements
An exciting feature of this project is the integration of liveness detection, which guards against facial spoofing attacks and ensures that the recognized face is genuinely present. Liveness detection helps determine if the face is from a live person or a fake representation.

Future enhancements will likely involve integrating liveness detection models, such as Convolutional Neural Networks (CNNs) or anti-spoofing algorithms. These models will analyze subtle facial cues and characteristics to distinguish between live faces and non-living artifacts.
