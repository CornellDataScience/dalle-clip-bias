from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route('/test')
def get_test():
    return {'output': 'test'}

@app.route('/clip')
def get_clip():
    prompt = request.args.get('prompt', 'test', type = str)
    outputImage = request.args.get('outputImage', './output.png', type = str)
    subprocess.run(['python3', 'generate.py', '-p', prompt, '-o', '../public/' + outputImage, '-i', '1'])
    return {'output': outputImage}