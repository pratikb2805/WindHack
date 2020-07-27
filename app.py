from flask import Flask, render_template, request, jsonify
from model import in_out
from datetime import datetime, timedelta

app = Flask('WindHack')
app.config['DEBUG'] = False

@app.route('/')
def homepage():
    return render_template('main.html')

@app.route('/view')
def index():
    return render_template( '/index.html') 

@app.route('/get', methods = ['GET', 'POST'])
def getGraph():
    nsteps = request.form['range']
    graph, maxPow, rangeNew = in_out(int(nsteps)*6)
    rangeNew = int(rangeNew)
    timeNow = datetime.now()
    timeNew = timeNow  + timedelta(minutes = 10*rangeNew)

    return jsonify({'graph': graph, 'maxPow' : maxPow,'time':timeNew })

if __name__ == '__main__':
    app.run()