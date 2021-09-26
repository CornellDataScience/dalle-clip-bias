from flask import Flask, request, render_template

app = Flask(__name__, static_url_path='')

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/image/", methods = ['GET', 'POST'])
def generateImage():
    # Use model
    #return request.args.get('prompt')
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)
