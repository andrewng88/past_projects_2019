from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

#to let Flask know that it can find the HTML template folder ( templates )
app = Flask(__name__)

@app.route('/') #execute this when home is triggered
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    SVM= open('SVM.pkl','rb')
    clf = joblib.load(SVM)

    if request.method == 'POST':
        message = request.form['message']
        cv=CountVectorizer()
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True) #activate Flask debugger