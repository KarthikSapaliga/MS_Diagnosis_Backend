from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from pkg.mri_inference import predict_volume
from pkg.oct_inference import predict_image

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'nii', 'nii.gz', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict/mri", methods=["POST"])
def predict_mri():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    fname, label, prob = predict_volume(file_path)

    response = {"mri_prob": prob, "mri_label": label}
    return jsonify(response)


@app.route("/predict/oct", methods=["POST"])
def predict_oct():
    mri_prob = float(request.form.get("mri_prob", 0))
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    prob, pred = predict_image(file_path)

    if mri_prob >= 0.2:
        final_prob = (mri_prob + prob) / 2
        final_prediction = "MS" if final_prob >= 0.5 else "NORMAL"
    else:
        final_prob = prob
        final_prediction = "MS" if prob >= 0.5 else "NORMAL"

    response = {
        "oct_prob": prob,
        "oct_pred": "MS" if pred == 1 else "NORMAL",
        "final_prediction": final_prediction,
        "combined_prob": final_prob
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
