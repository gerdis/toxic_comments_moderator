import numpy as np
import os
import re
import pandas as pd
import sqlite3 as sq
from dotenv import load_dotenv
from myutils import prepare, embed, single_prediction, givve_feedback
from keras.models import load_model
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, g
from flask_mail import Mail, Message
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, validators, SubmitField
from gensim.models import word2vec, KeyedVectors

#create app
app = Flask(__name__)

load_dotenv()

app.config['SECRET_KEY'] = os.environ.get("secret_key")
app.config['MAIL_SERVER']='smtp.mailtrap.io'
app.config['MAIL_PORT'] = 2525
app.config['MAIL_USERNAME'] = os.environ.get("mail_username")
app.config['MAIL_PASSWORD'] = os.environ.get("mail_password")
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

def load_cnn():
    """
    load pre-trained model
    """
    global model
    model = load_model('model_resources/New_toxic_CNN_2.h5')


class ContactForm(FlaskForm):
    """User entry form"""
    yourmail = StringField(description="Enter your email address", validators=[
                     validators.InputRequired(message="Please enter your email address"),
                     validators.regexp(r"^[A-Za-z0-9]{2,}([_.-][A-Za-z0-9]+)?@[-a-z0-9._]{2,}.[a-z]+",
                     message="Please enter a valid email address!")])
    subject = StringField(description="Enter a subject", validators=[
                     validators.InputRequired(message="Please enter a subject")])    
    mess = TextAreaField(description="Enter your message", validators=[
                     validators.InputRequired(message="Please enter a message")])
                        
    submit = SubmitField("Submit", id="sbn") 


  
class Member():
    def __init__(self, name, sql_data):
        self.name = name  
        conn = sq.connect(sql_data)       
        data = pd.read_sql("""SELECT * FROM members WHERE name = ?""", 
                           conn, params=[self.name])
        conn.close()
        self.profession = data['profession'].values[0]
        self.email = data['email'].values[0]       
        

#remaining attempts to send 'toxic' message
attempts = 3


@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')


@app.route("/<name>", methods=['GET', 'POST'])
def profile(name):
    try:
        person = Member(name, 'members.db')           
        global attempts        
        form = ContactForm()        
        if form.validate_on_submit():    
            yourmail = request.form['yourmail']
            subject = request.form['subject']
            mess = request.form['mess']
            processed_comment = prepare(mess)
            embedding = embed(processed_comment)
            prediction = single_prediction(embedding, model)[0]        
            if not np.any(prediction):        
                msg = Message(subject, sender=yourmail, recipients=[person.email])
                msg.body = mess
                mail.send(msg)
                thecolor = "black"
            else:        
                attempts-= 1
                thecolor = "red"        
            thefeedback = givve_feedback(prediction)      
            return render_template('response.html', name=person.name, prof=person.profession, 
                                   form=ContactForm(formdata=None), feedback=thefeedback, 
                                   color=thecolor, attempts=attempts)
        return render_template('profile.html', name=person.name, prof=person.profession, 
                               form=form, attempts=attempts)
    except IndexError:
        return f'''<h1> User '{name}' not found!</h1>'''
    

if __name__ == "__main__":
    print(("* Loading AI model resources and starting server..."
           "This process may take a few minutes"))    
    load_cnn()
    app.run("0.0.0.0", port=80)
