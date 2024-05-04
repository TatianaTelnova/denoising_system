from flask import Flask, render_template

app = Flask(__name__)

IMG_FOLDER = 'static/img/'
app.config["UPLOAD_FOLDER"] = IMG_FOLDER


@app.route('/')
def index():
    img = {'graph_result': app.config["UPLOAD_FOLDER"] + 'test.jpg'}
    return render_template('index.html', img=img)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3000, debug=True)
