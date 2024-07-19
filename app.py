import keras_ocr
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from PIL import Image
import pytesseract
from gtts import gTTS
# from ultralytics import YOLO
import cv2
# import cvzone
import math
import numpy as np




app = Flask(__name__)
def anotate(image_path='C://Users//jayag//Desktop//Marwadi//SEM-6//Mini-Project//3.jpeg'):
    # Create a pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Get an image
#     image_path = 'C://Users//jayag//Desktop//Marwadi//SEM-6//Mini-Project//3.jpeg'
    image = keras_ocr.tools.read(image_path)

    # Use the pipeline to extract text from the image
    prediction_groups = pipeline.recognize([image])

    # Plot the predictions
    fig, ax = plt.subplots()
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0], ax=ax)
#     print(type(keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0], ax=ax)))
    plt.savefig("static/result.jpg")
    
    
        # Extract Predictions
    prediction_groups = pipeline.recognize([image])

    # Group words by their y-coordinate to identify lines of text
    line_groups = {}
    for word_group in prediction_groups[0]:
        y_coordinate = round(word_group[1][0][1], 2)  # Round y-coordinate to handle floating point errors
        if y_coordinate in line_groups:
            line_groups[y_coordinate].append(word_group)
        else:
            line_groups[y_coordinate] = [word_group]

    # Sort the lines based on their y-coordinate
    sorted_lines = sorted(line_groups.items(), key=lambda item: item[0])

    # Extract words from sorted lines and combine them to form the final sequence
    detected_sequence = ''
    for _, line in sorted_lines:
        # Sort words within each line based on their x-coordinate
        sorted_words = sorted(line, key=lambda word_group: word_group[1][0][0])
        line_words = [word_group[0] for word_group in sorted_words]

        # Check text direction
        if len(line_words) >= 2 and sorted_words[0][1][0][0] > sorted_words[1][1][0][0]:
            # If the text is detected from right to left, reverse the order of words within the line
            line_words.reverse()

        line_text = ' '.join(line_words)
        detected_sequence += line_text + ' '

    # Print the final sequence
    print("Detected Sequence:", detected_sequence)
    return detected_sequence
print(anotate())


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_to_text', methods=['GET', 'POST'])
def image_to_text():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image found in request!"
        image = request.files['image']
        if image.filename == '':
            return "No image selected!"
        try:
            img = Image.open(image)
            img.save("static/original.jpg")
            result = pytesseract.image_to_string(img)
            return render_template('image_to_text.html', result=result)
        except Exception as e:
            return f"An error occurred: {str(e)}"
    else:
        return render_template('image_to_text.html')
    
    
@app.route('/image_to_speech', methods=['GET', 'POST'])
def image_to_speech():
    result=0
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image found in request!"

        image = request.files['image']
        if image.filename == '':
            return "No image selected!"

        try:
            img = Image.open(image)
            text = pytesseract.image_to_string(img)

            # Convert text to speech
            tts = gTTS(text=text, lang='en')
            tts.save("static/output.mp3")
            result=1
            # Return the path to the generated speech file
            return render_template('image_to_speech.html',result=result)
        except Exception as e:
            return f"An error occurred: {str(e)}"
    else:
        return render_template('image_to_speech.html')
    
# @app.route('/object_detection',methods=['GET','POST'])
# def object_detection():
#     result=0
#     model = YOLO("static/yolov8n.pt")
#     classnames = ['person','bicycle','car','motorcycle','airplane','bus','train','truck']
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return "No image found in request!"
#         image = request.files['image']
#         if image.filename == '':
#             return "No image selected!"
#         try:
#             img = cv2.imread(image)
#             results = model(img, stream=True)
#             detections = np.empty((0, 5))
#             for r in results:
#                 boxes=r.boxes
#                 for box in boxes:
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
#                     bbox=x1,y1,x2,y2
#                     conf = math.ceil((box.conf[0]*100))/100
#                     cls = box.cls[0]
#                     cvzone.cornerRect(img,bbox,l=9)
#                     cvzone.putTextRect(img, f'{classnames[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
#                     cv2.imwrite('static/detection.jpg', img)
#             return render_template('object_detection.html', result=1)
#         except Exception as e:
#             return f"An error occurred: {str(e)}"
#     else:
#         return render_template('object_detection.html') 

# if __name__ == '__main__':
#     app.run(debug=True,use_reloader=False)


