from flask import Flask, request
import os
import urllib.request
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
from ContractsOCR import ocrDoc
from pdfContract import extract_from_pdf


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['pdf'])
# UPLOADS_PATH = join(dirname(realpath(__file__)), 'uploads/..')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def home():
    print("hello world")
    return "Hello World"

@app.route('/contract-ocr', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if request.form.get("File"):
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files.get('File')
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        print(os.path.join(app.config['UPLOAD_FOLDER']))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
        resp = jsonify(
            {'message': 'File successfully uploaded :{}'.format(filename),'results':extract_from_pdf(filename)})
        # j=ocrDoc(filename)
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'] , filename)):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'] , filename))

        return resp
    else:
        resp = jsonify({'message': 'Allowed file type is pdf'})
        resp.status_code = 400
        return resp


if __name__ == '__main__':
    app.run(debug=True,threaded=True)
