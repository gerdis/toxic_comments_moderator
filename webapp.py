import numpy as np
import os
import re
from dotenv import load_dotenv
from myutils import preprocess, embed, single_prediction, give_feedback
from keras.models import load_model
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_mail import Mail, Message
from wtforms import Form, StringField, TextAreaField, validators, SubmitField
from gensim.models import word2vec, KeyedVectors

#create app
app = Flask(__name__)

load_dotenv()

app.config['MAIL_SERVER']='smtp.mailtrap.io'
app.config['MAIL_PORT'] = 2525
app.config['MAIL_USERNAME'] = os.environ.get("mail_username")
app.config['MAIL_PASSWORD'] = os.environ.get("mail_password")
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

class ReusableForm(Form):
    """User entry form"""
    yourmail = StringField("Your email: ", validators=[
                     validators.InputRequired(),
                     validators.regexp(r"^[A-Za-z0-9]{2,}([_.-][A-Za-z0-9]+)?@[-a-z0-9._]{2,}.[a-z]+")])
    subject = StringField("Subject: ", validators=[
                     validators.InputRequired()])    
    mess = TextAreaField("Your message: ", validators=[
                     validators.InputRequired()])
                        
    submit = SubmitField("Submit")

def load_cnn():
    """
    load pre-trained model
    """
    global model
    model = load_model('model_resources/toxic_CNN.h5')
    
# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)
    
    if request.method == 'POST' and form.validate():
        # Extract information
        mess = request.form['mess']
        subject = request.form['subject']
        yourmail = request.form['yourmail']
        processed_comment = preprocess(mess)
        embedding = embed(processed_comment)
        prediction = single_prediction(embedding, model)[0]
        #Generate reply
        if not np.any(prediction):            
            msg = Message(subject, sender=yourmail, recipients=['johndoe@mailtrap.io'])
            msg.body = mess
            mail.send(msg)       
        return render_template('feedback.html', input=give_feedback(prediction, mess))        
    
    return render_template('index.html', form=form)

if __name__ == "__main__":
    print(("* Loading AI model resources and starting server..."
           "This process may take a few minutes"))    
    load_cnn()
    app.run("0.0.0.0", port=80)
