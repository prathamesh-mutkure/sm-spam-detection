from flask import Flask, request, jsonify, render_template, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import pandas as pd
import numpy as np

from sklearn import model_selection
from scipy import stats
import joblib

import io


app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database file path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret'

db = SQLAlchemy(app)

# Engine to connect to your database
engine = create_engine('sqlite:///mydatabase.db', echo=True)

# Session factory to manage interactions with the database
# Session = sessionmaker(bind=engine)

# Base class for all models
Base = declarative_base()


# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'
    

Base.metadata.create_all(engine)

fb_svm_model = joblib.load("facebook/fb_svm_model.sav")
fb_knn_model = joblib.load("facebook/fb_knn_model.sav")
fb_dt_model = joblib.load("facebook/fb_dt_model.sav")

insta_svm_model = joblib.load("insta/insta_svm_model.sav")
insta_knn_model = joblib.load("insta/insta_knn_model.sav")
insta_dt_model = joblib.load("insta/insta_dt_model.sav")

twt_svm_model = joblib.load("twt/twt_svm_model.sav")
twt_knn_model = joblib.load("twt/twt_knn_model.sav")
twt_dt_model = joblib.load("twt/twt_dt_model.sav")
twt_rf_model = joblib.load("twt/twt_rf_model.sav")


@app.route("/hello", methods=['GET'])
def hello():
    return "Hello World", 200


@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")


# Register route
@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')

    # Check if the username already exists
    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists'}), 409

    # Create a new user
    user = User(username=username, password=password)
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201

# Login route
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    # Find the user by username
    user = User.query.filter_by(username=username).first()

    # Check if the user exists and the password matches
    if user and user.password == password:
        return jsonify({'message': 'Login successful'})

    return jsonify({'message': 'Invalid credentials'}), 401

# Forgot password route
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    username = request.json.get('username')
    new_password = request.json.get('new_password')

    # Find the user by username
    user = User.query.filter_by(username=username).first()

    # Update the user's password
    if user:
        user.password = new_password
        db.session.commit()
        return jsonify({'message': 'Password updated successfully'})

    return jsonify({'message': 'User not found'}), 404


@app.route('/facebook.html', methods=['GET', 'POST'])
def fb_predict():
    if request.method == 'POST':

        file = request.files['file']
        df_full = pd.read_csv(file)

        selected_option = request.form['model']

        # df_full = pd.read_csv('facebook/facebook-dataset.csv')
        df = df_full

        df['Label'].replace([0,1], ['Legitimate', 'Fake'], inplace=True)
        df.drop('profile id', axis=1, inplace=True)

        df['Label'] = pd.Categorical(df['Label']).codes

        if selected_option == "svm":
            output = fb_svm_model.predict(df.drop(["Label"], axis=1)).tolist()
        elif selected_option == "knn":
            output = fb_knn_model.predict(df.drop(["Label"], axis=1)).tolist()
        else:
            output = fb_dt_model.predict(df.drop(["Label"], axis=1)).tolist()
            

        df.loc[:, "Label"] = output;
        df['Label'].replace([0,1], ['Legitimate', 'Fake'], inplace=True)

        df = df.rename(columns={'Label': selected_option})

        df.to_csv('/tmp/result.csv')

        return render_template("facebook.html", data=df.to_html())
    

    return render_template("facebook.html")
    

    # df = pd.read_csv('facebook/facebook-dataset.csv')
    # df['Label'].replace([0,1], ['Legitimate', 'Fake'], inplace=True)
    # df.drop('profile id', axis=1, inplace=True)
    # df['Label'] = pd.Categorical(df['Label']).codes
    # Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(df.drop(['Label'], axis=1), df["Label"], train_size=0.8)

    # File from frontend
    # file = request.files['file']

    # # Convert to DataFrame
    # df = pd.read_csv(file)

    # # Perform Prediction
    # output = fb_svm_model.predict(df).tolist()

    # print(output);

    # return jsonify({'message': output}), 200


@app.route('/instagram.html', methods=['GET', 'POST'])
def insta_predict():
    if request.method == 'POST':

        file = request.files['file']
        df_full = pd.read_csv(file)

        selected_option = request.form['model']

        df = df_full

        df['fake'].replace([0,1], ['Legitimate', 'Fake'], inplace=True)
        df['fake'] = pd.Categorical(df['fake']).codes

        if selected_option == "svm":
            output = insta_svm_model.predict(df.drop(["fake"], axis=1)).tolist()
        elif selected_option == "knn":
            output = insta_knn_model.predict(df.drop(["fake"], axis=1)).tolist()
        else:
            output = insta_dt_model.predict(df.drop(["fake"], axis=1)).tolist()
            

        df.loc[:, "fake"] = output;
        df['fake'].replace([0,1], ['Legitimate', 'Fake'], inplace=True)

        df = df.rename(columns={'fake': selected_option})

        df.to_csv('/tmp/result.csv')

        return render_template("instagram.html", data=df.to_html())
    

    return render_template("instagram.html")


@app.route('/twitter.html', methods=['GET', 'POST'])
def twt_predict():
    if request.method == 'POST':

        file = request.files['file']
        df_full = pd.read_csv(file)

        selected_option = request.form['model']

        df = df_full

        print(df)

        df['Label'].replace([0,1], ['Legitimate', 'Fake'], inplace=True)
        df['Label'] = pd.Categorical(df['Label']).codes

        if selected_option == "svm":
            output = twt_svm_model.predict(df.drop(["Label"], axis=1)).tolist()
        elif selected_option == "knn":
            output = twt_knn_model.predict(df.drop(["Label"], axis=1)).tolist()
        elif selected_option == "rf":
            output = twt_rf_model.predict(df.drop(["Label"], axis=1)).tolist()
        else:
            output = twt_dt_model.predict(df.drop(["Label"], axis=1)).tolist()
            

        df.loc[:, "Label"] = output;
        df['Label'].replace([0,1], ['Legitimate', 'Fake'], inplace=True)

        df = df.rename(columns={'Label': selected_option})

        df.to_csv('/tmp/result.csv')

        return render_template("twitter.html", data=df.to_html())
    

    return render_template("twitter.html")


@app.route('/download')
def download():
    file_path = '/tmp/result.csv'
    return send_file(file_path, as_attachment=True)


@app.route('/<template_name>', methods=['GET'])
def render_custom_template(template_name):
    print(template_name)
    return render_template(template_name)


if __name__ == '__main__':
    app.run(debug=True)
