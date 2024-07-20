import subprocess
import os
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

model_path = "yolov5/YOLOv5s_noAugmentation.pt"


def run_yolov5_inference(input_path):
    command = [
        "python",
        "yolov5/detect.py",
        "--source", input_path,
        "--weights", model_path
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    process.wait()

    return process.returncode


def find_output_file(input_path):
    dirname, filename = os.path.split(input_path)
    filename_without_ext, ext = os.path.splitext(filename)
    output_dir = os.path.join("yolov5","runs", "detect")

    # Find the latest experiment folder (expX) in the output_dir
    experiment_folders = [f for f in os.listdir(output_dir) if f.startswith("exp")]
    experiment_folders.sort(reverse=True)
    for exp_folder in experiment_folders:
        exp_folder_path = os.path.join(output_dir, exp_folder)
        if os.path.isdir(exp_folder_path):
            output_file_path = os.path.join(exp_folder_path, filename_without_ext + ext)
            if os.path.exists(output_file_path):
                return output_file_path

    return None


@app.route('/predict', methods=['POST'])
def predict():
    if 'input' not in request.files:
        return jsonify({'error': 'No file provided'})

    input_file = request.files['input']
    input_filename = input_file.filename

    temp_path = os.path.join(".", "temp_file" + os.path.splitext(input_filename)[1])
    input_file.save(temp_path)

    is_image = any(ext in input_filename.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'])
    is_video = any(ext in input_filename.lower() for ext in ['.mp4', '.avi', '.mov'])

    if is_image or is_video:
        return_code = run_yolov5_inference(temp_path)
    else:
        # Invalid file format
        os.remove(temp_path)
        return jsonify({'error': 'Invalid file format. Supported formats are image and video files.'})

    if return_code == 0:
        output_file = find_output_file(temp_path)
        os.rename(output_file, "out/" + output_file.split("/")[-1])
        output_file = "out/" + output_file.split("/")[-1]

        if output_file:
            os.remove(temp_path)
            if is_image:
                return send_file(output_file, mimetype='image/jpeg')
            elif is_video:
                return send_file(output_file, mimetype='video/mp4')
        else:
            return jsonify({'error': 'Output file not found'})

    return jsonify({'error': 'YOLOv5 inference failed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
