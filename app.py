'''
from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet,configure_uploads,ALL,DATA

# we have to be able to secure the file so lets import the package below
from werkzeug import secure_filename

app = Flask(__name__)
Bootstrap(app)

#configuration
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadstorage'
configure_uploads(app,files)



import os
import datetime
import time

# EDA packages
import pandas as pandas
import numpy as np 

# ML packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ml packages for vectorization and feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

@app.route('/')
def index():
	return render_templete('index.html')


@app.route('/datauploads',methods=['GET','POST'])
# we had a method called post so we have to call the methosd here
def datauploads():
	if request.method == 'POST' and 'csv_data' in request.file:
	 # csv_data is coming from the the html file
	 	file= request.files['csv_data']
	 	filename = secure_filename(file.filename)
	 	# lets be able to save the file in our static folder
	 	file.save(os.path.join('static/uploadstorage', filename))


	 	#date 
	 	date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

	 	# EDA functions
	 	df =  pd.read_csv(os.path.join('static/uploadstorage', filename))
	 	df_size = df.size 
	 	df_shape = df.shape 
	 	df_columns = list(df.columns)
	 	# df.loc[:, df.columns ! = 'b']
	 	df_targetname = df[df.columns[-1]].name 
	 	df_featurenames = df_columns[0: 1]
	 	# select all columns till last column
	 	df_xfeatures = df.iloc[:, 0: -1]
	 	# select the last column as target
	 	df_ylabels= df[df.columns[-1]]
	 	#df_ylabels= df.iloc[:,-1]


	 	# lets create a tabel so we will able to read the data
	 	df_table = df


	 	# so lets get our x and y
	 	x= df_xfeatures
	 	y =df_ylabels

	 	# models building
	 	# to wrok on our model we to include our cels
	 	models = []
	 	models.append(('LR', LogisticRegression()))
	 	models.append(('LDA', LinearDiscriminantAnalysis()))
	 	models.append(('KNN', KNeighborsClassifier()))
	 	models.append(('CART', DecisionTreeClassifier()))
	 	models.append(('NB', GaussianNB()))
	 	models.append(('SVM', SVC()))

	 	results = [] 
	 	names = []
	 	allmodels = []
	 	scoring = 'accuracy'

	 	for name, model in models:
	 		kfold =model_selection.KFold(n_splits=10, random_state=seed)
	 		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)# cv=kfold,
	 		results.append(cv_results)
	 		names.append(name)
	 		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	 		allmodels.append(msg)
	 		model_results = results
	 		model_names = names


	return render_templete('details.html', filename=filename, df_table = df,
		date=date,
	 	df_shape=df_shape, 
	 df_columns=df_columns,
	  df_targetname=df_targetname,
	 model_results=allmodels,
	  model_names=names,
	  fullfile=fullfile, )


	if __name__=='__main__':
		app.run(debug=True)

		'''




		# Flask Packages
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy 

from werkzeug import secure_filename
import os
import datetime
import time


# EDA Packages
import pandas as pd 
import numpy as np 

# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer




app = Flask(__name__, template_folder='template')
Bootstrap(app)
db = SQLAlchemy(app)

# Configuration for File Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadstorage'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadstorage/filestorage.db'

# Saving Data To Database Storage
class FileContents(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	name = db.Column(db.String(300))
	modeldata = db.Column(db.String(300))
	data = db.Column(db.LargeBinary)


@app.route('/')
def index():
	return render_template('index.html')



# Route for our Processing and Details Page
@app.route('/dataupload',methods=['GET','POST'])
def dataupload():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
		# os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('static/uploadstorage',filename))
		fullfile = os.path.join('static/uploadstorage',filename)

		# For Time
		date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

		# EDA function
		df = pd.read_csv(os.path.join('static/uploadstorage',filename))
		df_size = df.size
		df_shape = df.shape
		df_columns = list(df.columns)
		df_targetname = df[df.columns[-1]].name
		df_featurenames = df_columns[0:-1] # select all columns till last column
		df_Xfeatures = df.iloc[:,0:-1] 
		df_Ylabels = df[df.columns[-1]] # Select the last column as target
		# same as above df_Ylabels = df.iloc[:,-1]
		

		# Model Building
		X = df_Xfeatures
		Y = df_Ylabels
		seed = 7
		# prepare models
		models = []
		models.append(('LR', LogisticRegression()))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))
		# evaluate each model in turn
		

		results = []
		names = []
		allmodels = []
		scoring = 'accuracy'
		for name, model in models:
			kfold = model_selection.KFold(n_splits=10, random_state=seed)
			cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			allmodels.append(msg)
			model_results = results
			model_names = names 
			
		# Saving Results of Uploaded Files  to Sqlite DB
		newfile = FileContents(name=file.filename,data=file.read(),modeldata=msg)
		db.session.add(newfile)
		db.session.commit()		
		
	return render_template('details.html',filename=filename,date=date,
		df_size=df_size,
		df_shape=df_shape,
		df_columns =df_columns,
		df_targetname =df_targetname,
		model_results = allmodels,
		model_names = names,
		fullfile = fullfile,
		dfplot = df
		)




if __name__ == '__main__':
	app.run(debug=True)
