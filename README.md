A simple web application demonstrating how a sentiment analysis model can be deployed as an email moderation/filtering system.

# Background

The model, a Convolutional Neural Network, was trained by myself on the *toxic comments* dataset using Google's pre-trained
300-dimensional vector embeddings. 
GlobalMaxPooling and a generator were used so that sequences of varying length could be processed without padding. 

# Instructions for testing 

If you'd like to test the application yourself, you'll need the "GoogleNews" vectors (GoogleNews-vectors-negative300.bin). 
You can download them from [Kaggle](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300), 
although I'm not sure if it's the same version that I used for training 
(obviously, if it's a different version, this might affect model performance).  

The following Python packages are required:

- numpy
- pandas
- nltk (for natural language processing)
- keras (to load pre-trained model)
- tensorflow (keras backend)
- gensim (to map words to google's pre-trained vectors)
- wtforms (to build the form for the message)
- flask (to build web applications in Python)
- flask-wtf (form handling extension)
- flask-mail (email extension)
- os + python-dotenv (if you'd like to store your email credentials as environment variables)

The next steps are as follows:

1. clone the repository
2. save the binary file with the google vectors in the *model_resources* folder
3. replace the email configuration details/credentials in webapp.py with your own
4. create a secret key for CSRF protection, and replace the one in webapp.py with your own

If you'd like to use [Mailtrap](https://mailtrap.io/email-sandbox/) for email testing (as I did), 
you can find an instruction on how to get your configuration details and credentials 
[here](https://mailtrap.io/blog/flask-email-sending/).

Then run 

```python webapp.py```

and open *http://127.0.0.1* in a browser. Select one of the people listed under "current members" (currently, there are only
2 members stored in the database *members.db*). Complete the contact form on their profile page with your email address, 
a subject and a message, and click "submit". 
If your message is classified as "toxic", you'll receive a notification that the message could not be sent. 
The application will also inform you which types/severity levels of "toxicity" 
(e.g. "general toxicity" for label *toxic*) were detected.
Otherwise, you should receive a notification that your message was sent, and find the email in your inbox.

If you need to get a lot of "toxicity" out of your system, feel free to try sending multiple nasty messages - however, 
you shouldn't be able to submit more than 3 such messages. Repeat offenders will have the form fields and submit button disabled.

# References

Konstantinos Sechidis, Grigorios Tsoumakas, and Ioannis P. Vlahavas. 2011. On the Stratification of Multi-label Data. 
In *Proceedings of the Machine Learning and Knowledge Discovery in Databases – European Conference, ECML PKDD (part III)*, 
pages 145–158, Athens, Greece.

# Credits

[This tutorial by Will Koehrsen](https://towardsdatascience.com/deploying-a-keras-deep-learning-model-as-a-web-application-in-p-fc0f2354a7ff) 
helped me greatly in getting the app up and running.

## Photo credits

- [Customer icon by Icons8](https://icons8.com/icon/14736/customer)

- website background: photo by Steve Johnson
