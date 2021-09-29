from flask import Flask, request, render_template
import subprocess

app = Flask(__name__, static_url_path='')

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/image/", methods = ['GET', 'POST'])
def generateImage():
    prompt = request.args.get("prompt", default = "student", type = str)
    fileName = prompt+".png"
    subprocess.run(["python", "/home/ubuntu/dalle-clip-bias/CLIP+Optim/generate.py", "-p", prompt, "-o", "/home/ubuntu/dalle-clip-bias/templates/static/" + fileName])
    return render_template("index.html", user_image = fileName)
    
if __name__ == "__main__":
    app.run(debug=True)
