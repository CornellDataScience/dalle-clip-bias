from flask import Flask
import subprocess

app = Flask(__name__)

@app.route('/test')
def get_test():
    return {'output': 'test'}

@app.route('/clip')
def get_clip():
    outputImage = './output.png'
    subprocess.run(['python3', 'generate.py', '-p', 'test', '-o', '../public/' + outputImage, '-i', '1'])
    return {'output': outputImage}