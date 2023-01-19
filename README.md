# A simple web application demonstrating how a sentiment analysis model can be deployed as an email moderation/filtering system.

# Background

The model, a Convolutional Neural Network, was trained on the ***toxic comments*** dataset using Google's pre-trained
300-dimensional vector embeddings. 
During training, a Keras generator was used to generate batches so that no padding was needed. 

The code for the model and cross validation results can be found in ***ToxicCommentsProjectCrossVal.ipynb*** 
(I might re-train the model using a slightly different architecture).

# Instructions for testing 

If you'd like to test the application yourself, you'll need the "GoogleNews" vectors (GoogleNews-vectors-negative300.bin). 
You can download them from [Kaggle](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300), 
although I'm not sure if it's the same version that I used for training 
(obviously, if it's a different version, this might affect model performance).  

The following Python packages are required:

numpy
keras (to load pre-trained model)
tensorflow (keras backend)
gensim (to map words to google's pre-trained vectors)
wtforms (to build the form for the message)
flask (to build web applications in Python)
flask-mail (email extension)
os + python-dotenv (if you'd like to store your email credentials as environment variables)

The next steps are as follows:

1. clone the repository
2. save the binary file with the google vectors in the ***model_resources*** folder
3. replace the email configuration details/credentials in webapp.py with your own 

If you'd like to use [Mailtrap](https://mailtrap.io/email-sandbox/) for email testing (as I did), 
you can find an instruction on how to get your configuration details and credentials 
[here] (https://mailtrap.io/blog/flask-email-sending/).

Then run 

```python webapp.py```

and open the link in a browser. Note that I tested the applicatin with firefox
and didn't spend much time fine-tuning the CSS, so the design might not look as expected in other browsers.

After opening the link, simply enter an email address, a subject and a message, and click "submit". 
If your message is classified as "toxic", you'll receive a notification that the message could not be sent. 
The application will also inform you which types/severity levels of "toxicity" 
(e.g. "general toxicity" for label ***toxic***) were detected.
Otherwise, you should receive a "Thank you for your message", and find the email in your inbox. 

# Photo Credits

- [Customer icon by Icons8] (https://icons8.com/icon/14736/customer)

- website background: photo by Steve Johnson
