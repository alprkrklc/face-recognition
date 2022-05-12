from face_rec import Recognizator

# Path to your folder loaded with face images so program can recognize them in other images.
KNOWN_FACES_PATH = './known_faces'
TEST_IMAGE = 'test.jpg'

def main():
    rec = Recognizator(KNOWN_FACES_PATH)
    
    faces = rec.get_faces(TEST_IMAGE, display_image=True)
    names = [face.name for face in faces]
    
    print(f'Face(s) in {TEST_IMAGE}: {", ".join(names)}.')

if __name__ == '__main__':
    main()
