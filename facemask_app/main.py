import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from faster_rcnn_mask_detector import FaceMaskDetector

INPUT_FOLDER = '../media/input/'
OUTPUT_FOLDER = '../media/output/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['INPUT_FOLDER'] = INPUT_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * (1024 * 1024)  # 16 MB


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_face_mask(input_file_path):
    detector = FaceMaskDetector(num_classes=3)
    output_file_path = detector.detect(input_file_path)
    return output_file_path


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['INPUT_FOLDER'], filename))

            input_file_path = os.path.join(
                app.config['INPUT_FOLDER'], filename)
            output_file_path = detect_face_mask(input_file_path)
            output_filename = os.path.basename(output_file_path)

            return redirect(url_for('uploaded_file',
                                    filename=output_filename))
    return '''
    <!doctype html>
    <title>Detect Face Mask</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Detect Face Mask>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
