import json
from application import app, Recognition, mysql, get_response
from flask import render_template, request, Response, jsonify, Blueprint, redirect, session, flash, url_for
from functools import wraps
import numpy as np
from cv2 import CAP_DSHOW, VideoCapture
from flask_cors import CORS
import MySQLdb.cursors

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")
def gen(__init__):
    while True:
        frame = __init__.gen_frames()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Recognition()), mimetype='multipart/x-mixed-replace; boundary=frame')

#check if user logged in
def is_logged_in(f):
	@wraps(f)
	def wrap(*args,**kwargs):
		if 'logged_in' in session:
			return f(*args,**kwargs)
		else:
			flash('Unauthorized, Please Login','danger')
			return redirect(url_for('login'))
	return wrap


@app.route("/absensi")
@is_logged_in
def absensi():
    return render_template("absensi.html")

@app.route("/profil",methods=['POST','GET'])
@is_logged_in
def profil():
    return render_template("profil.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/help")
def help():
    return render_template("help.html")
@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

@app.route("/login",methods=['POST','GET'])
def login():
    status=True
    if request.method=='POST' and 'username' in request.form and 'password' in request.form:
        username=request.form["username"]
        pwd=request.form["password"]
        
        cur=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("select * from users where username=%s and password=%s",(username,pwd))

        data=cur.fetchone()
        if data:
            session['logged_in']=True
            session['username']=data["fullname"]
            session['fullname']=data["username"]
            session['ttl']=data["ttl"]
            session['alamat']=data["alamat"]
            # session['gambar']=data["gambar"]
            flash('Login Successfully','success')
            return redirect(url_for('index'))
        else:
            flash('Invalid Login. Try Again','danger')
    return render_template("login.html",status=status)

#logout
@app.route("/logout")
def logout():
	session.clear()
	flash('You are now logged out','success')
	return redirect(url_for('login'))

if __name__ == "__main__":
    # gunicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
    app.run(debug=True)

