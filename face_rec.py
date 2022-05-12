import os
import cv2
import string
import numpy as np
from PIL import Image
import face_recognition as fr

# Supported image extensions.
IMAGE_EXTENSIONS = ['.jpg', '.png']

# Label for unknown faces.
UNKNOWN_FACE_LABEL = 'Unknown'

class Face:
    '''
    A helper class to work with faces.
    '''
    def __init__(self, name: str, encoding: np.ndarray):
        self.name = name
        self.encoding = encoding

class Recognizator:
    '''
    Face recognition class.
    '''
    def __init__(self, known_faces_path):
        '''
        :param known_faces_path: Path to faces folder.
        '''
        self._faces = []
        self.known_faces_path = known_faces_path

    def get_known_faces(self, force_reload=False) -> list[Face]:
        '''
        Loads and returns the known faces.
        If faces are already loaded, it'll return it from cache.
        
        :param force_load: Even if faces are already loaded, it'll force program to reload them.
        :return: List of `Face` objects.
        '''
        if force_reload or not self._faces:
            self.load_faces()
        
        return self._faces

    def load_faces(self):
        '''
        Looks through `known_faces_path` directory, encodes all the faces and saves them to cache.
        '''
        self._faces.clear()
        
        for root, dirs, files in os.walk(self.known_faces_path):
            for file in files:
                # If file extension is not supported, skips to the next file.
                name, ext = os.path.splitext(file)
                if ext not in IMAGE_EXTENSIONS:
                    continue
                
                # Loads the image.
                face = fr.load_image_file(os.path.join(self.known_faces_path, file))

                # Encodes the face in that image.
                # We'll be using [0] because `fr.face_encodings` returns list of all faces' encodings in the image. 
                encoding = fr.face_encodings(face)[0]
                
                # Adds encoded faces to list.
                self._faces.append(Face(string.capwords(name), encoding))

    def get_faces(self, image_path, display_image=True) -> list[Face]:
        '''
        Finds all the faces in a given image and returns list of `Face` objects.

        :param image_path: Path of the image to try to recognize faces in it.
        :param display_image: After finding faces in given image, you can display the image with faces marked.
        :return: List of `Face` objects of given image.
        '''
        known_faces = self.get_known_faces()
        known_encodings = [face.encoding for face in known_faces]

        # Reads given image.
        image = cv2.imread(image_path, 1)

        # Locates all faces in given image.
        face_locations = fr.face_locations(image)
        
        # Gets encodings of all faces in given image.
        unkown_encodings = fr.face_encodings(image, face_locations)
        
        faces = []
        for encoding in unkown_encodings:
            # Compares if there is a match between known faces and current face from given image.
            matches = fr.compare_faces(known_encodings, encoding)

            # Selects the known face with the minimum distance to the current face from given image.
            face_distances = fr.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(face_distances)

            # Crates or gets the face instance.
            face = Face(UNKNOWN_FACE_LABEL, encoding)
            if matches[best_match_index]:
                face = known_faces[best_match_index]
                face.encoding = encoding

            faces.append(face)
        
        if display_image:
            self.display(image, face_locations, faces, image_path)
        
        return faces
    
    def display(self, image, face_locations, faces, window_title):
        '''
        Displays the given image with faces marked.

        :param image: Output of `cv2.imread` function.
        :param face_locations: Output of `fr.face_locations` function.
        :param faces: List of `Face` objects.
        :param window_title: Title of the image.
        '''
        for (top, right, bottom, left), face in zip(face_locations, faces):
            # Draws a box around the face.
            cv2.rectangle(image, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draws a label with the name of the face.
            cv2.rectangle(image, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, face.name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

        # Converts cv2 image to color array.
        color_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Converts color array to Pillow image.
        image = Image.fromarray(color_array)

        # Displays the image.
        image.show(window_title)
