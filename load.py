import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR, NuSVR

def print_scores(model, model_type=''):
	train_score = model.score(X_train, y_train)
	test_score = model.score(X_test, y_test)
	print(model_type + " Training score: {:.2f}".format(train_score))
	print(model_type + " Testing score: {:.2f}".format(test_score))
	#print(model_type + " Number of features used: {}".format(np.sum(model.coef_ != 0)))
	print()

def graph_ridge(ridge, ridge10, ridge01, lr):
	plt.plot(ridge.coef_, 's', label='Ridge alpha=1')
	plt.plot(ridge10.coef_, '^', label='Ridge alpha=10')
	plt.plot(ridge01.coef_, 'v', label='Ridge alpha=.1')

	plt.plot(lr.coef_, 'o', label='Linear Regression')
	plt.xlabel("Coefficient index")
	plt.ylabel("Coefficent magnitude")
	plt.hlines(0, 0, len(lr.coef_))
	plt.legend()
	plt.show()


def spider(url, file):
	fw = open(file, 'w')
	source_code = requests.get(url)  # connects to url and stores info in variable source_code
	plain_text = source_code.text  # takes text of request and stores it in plain_text (links, images etc.)
	soup = BeautifulSoup(plain_text, 'html.parser')  # soup object is what you sort through
	# for each row in the table
	for th in soup.find_all('th'):
		text = th.text
		fw.write(text + ',')
		print(th.string, '', end='')
	fw.write('\n')
	count = 0
	for tr in soup.find_all('tr'):
		count += 1
		if not tr:
			continue
		# for each entry in a row
		for td in tr.find_all('td'):
			if not td:
				continue
			# print each entry in that row
			text = td.text
			text = text.replace(',', '')
			fw.write(text + ',')
			print(td.string, '', end='')
		fw.write('\n')
		print('')
	fw.close()


def grab_data():
	url = 'https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=4&stype=0&sn=3&scope=0&w=0&ew=0&s=&t=0&p=2&st=FantasyPointsYahoo&d=1&ls=FantasyPointsYahoo&live=false&pid=false&minsnaps=4'
	output_file = 'qb_points_2014.csv'
	spider(url, output_file)

def load_test(qb_file, team_off_file):
	qb_points = pd.read_csv(qb_file)
	team_off = pd.read_csv(team_off_file)

	targets = qb_points['Fantasy Points']

	qb_points = qb_points.drop('Fantasy Points', axis=1)
	qb_points = pd.merge(qb_points, team_off, on='Team')
	qb_points = qb_points.drop('Rk', axis=1)
	qb_points = qb_points.drop('Pos', axis=1)
	qb_points = qb_points.drop('PPG',axis=1)
	qb_points = qb_points.drop('Team', axis=1)

	names = qb_points.ix[1]

	qb_points = qb_points.drop('Player', axis=1)
	#qb_points.to_csv('test_data.csv', sep=',')

	return qb_points, np.array(targets), names


def load_data(prev_year_file, next_year_file, team_off_file):
	prev_year = pd.read_csv(prev_year_file, header=0)
	next_year = pd.read_csv(next_year_file, header=0)
	team_off = pd.read_csv(team_off_file, header=0)

	names = prev_year.ix[:,0]

	col_list = ['Player', 'Fantasy Points']
	next_year = next_year[col_list]

	prev_year = prev_year.drop('Fantasy Points', axis=1)
	prev_year = pd.merge(prev_year, team_off, on='Team')
	prev_year = pd.merge(prev_year, next_year, on='Player')
	#prev_year.to_csv('data.csv', sep=',')

	prev_year = prev_year.drop('Rk', axis=1)
	prev_year = prev_year.drop('Pos', axis=1)
	prev_year = prev_year.drop('PPG',axis=1)
	prev_year = prev_year.drop('Team', axis=1)

	prev_year = prev_year.drop('Player', axis=1)

	targets = prev_year.ix[:,-1]
	#print("{}".format(prev_year))
	prev_year = prev_year.drop('Fantasy Points', axis=1)

	features = list(prev_year.columns.values)
	#del features[0]

	#data = prev_year._get_numeric_data()
	#print("{}".format(prev_year))
	#data = prev_year.as_matrix()

	return names, features, prev_year, np.array(targets)


def print_info():
	print("Targets: ".format(y))
	print("Y train:{}".format(y_train))
	print("Y test:{}".format(y_test))
	print("X train:{}".format(X_train))
	print("X test:{}".format(X_test))	
	print("Features: {}".format(features))


def scores():
	print_scores(lr,'Linear Regression')
	print_scores(ridge,'Ridge')
	print_scores(ridge01,'Ridge 01')
	print_scores(ridge10,'Ridge 10')
	print_scores(lasso,'Lasso')
	print_scores(lasso001,'Lasso 001')
	print_scores(lasso00001,'Lasso 00001')
	print_scores(svr,'SVR')
	print_scores(lin_svr,'LinSVR')
	print_scores(nu_svr,'NuSVR')

def predict():
	matt = np.array([16, 415, 628, 66, 4694, 8, 28, 14, 94, 29, 145, 5, 0, -0.073, 10, -0.154, 28, 0.028, 23, -0.154, 25, -0.033, 0.09, -0.144, 0.05, 10, 0.037, 29
]).reshape(1, -1)
	# matt ryan's 2014 stats
	#matt = np.array([16, 415, 628, 66, 4694, 8, 28, 14, 94, 29, 145, 5, 0]).reshape(1, -1)
	print("Matt 2015 LR Prediction: {}".format(lr.predict(matt)))
	print("Matt 2015 Ridge Prediction: {}".format(ridge.predict(matt)))
	print("Matt 2015 Ridge10 Prediction: {}".format(ridge10.predict(matt)))
	print("Matt 2015 Ridge01 Prediction: {}".format(ridge01.predict(matt)))
	print("Matt 2015 Lasso Prediction: {}".format(lasso.predict(matt)))
	print("Matt 2015 Lasso 001 Prediction: {}".format(lasso001.predict(matt)))
	print("Matt 2015 Lasso 00001 Prediction: {}".format(lasso00001.predict(matt)))
	print("Matt 2015 SVR Prediction: {}".format(svr.predict(matt)))
	print("Matt 2015 LinSVR 001 Prediction: {}".format(lin_svr.predict(matt)))
	print("Matt 2015 NuSVR 00001 Prediction: {}".format(nu_svr.predict(matt)))
	print("Matt 2015 Actual: 247.94")


def make_models(X_train, y_train):
	lr = LinearRegression().fit(X_train, y_train)
	ridge = Ridge().fit(X_train, y_train)
	ridge10 = Ridge(alpha=10, max_iter=1000000).fit(X_train, y_train)
	ridge01 = Ridge(alpha=.1, max_iter=1000000).fit(X_train, y_train)

	lasso = Lasso().fit(X_train, y_train)
	lasso001 = Lasso(alpha=.01, max_iter=1000000).fit(X_train, y_train)
	lasso00001 = Lasso(alpha=.0001, max_iter=1000000).fit(X_train, y_train)

	svr = SVR(C=100).fit(X_train, y_train)
	lin_svr = LinearSVR(C=100).fit(X_train, y_train)
	nu_svr = NuSVR(C=100).fit(X_train, y_train)

	return lr, ridge, ridge10, ridge01, lasso, lasso001, lasso00001, svr, lin_svr, nu_svr


def load_file(file):
	df = pd.read_csv(file)

	names = df['Player']
	cols = [1,2,3,4]
	y = df['Fantasy Points']

	df = df.drop(df.columns[cols], axis=1)
	df = df.drop('Fantasy Points', axis=1)
	df = df.drop('PPG', axis=1)
	return df, y, names


def write_file(file1, file2):
	df1 = pd.read_csv(file1)
	df2 = pd.read_csv(file2)

	tmp = ['Player', 'Fantasy Points']
	tmp_df = pd.DataFrame(df1[tmp])

	df1 = df1.drop('Fantasy Points', axis=1)
	df1 = pd.merge(df1, df2, on='Team')
	df1 = pd.merge(df1, tmp_df, on='Player')
	#df1.to_csv('test_data_2014.csv')


def get_raw(file):
	df = pd.read_csv(file)
	cols = [1,3,4,19]
	df = df.drop(df.columns[cols], axis=1)
	return df
#write_file('qb_points_2014.csv', 'team_offense_2014.csv' )

#grab_data()
#names, features, X_train, y_train = load_data('qb_points_2015.csv', 'qb_points_2016.csv', 'team_offense_2015.csv')
#X_tmp, y_test, names = load_test('qb_points_2014.csv', 'team_offense_2014.csv')

X_train, y_train, train_names = load_file('data.csv')
X_test, y_test, test_names = load_file('test_data_2014.csv')

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.fit_transform(X_test)

players = get_raw('test_data_2014.csv')

lr, ridge, ridge10, ridge01, lasso, lasso001, lasso00001, svr, lin_svr, nu_svr = make_models(X_train, y_train)
models = [lr, ridge, ridge10, ridge01, lasso, lasso001, lasso00001, svr, lin_svr, nu_svr]


i = 0
for model in models:
	for index, row in players.iterrows():
		player = row.drop(['Player','Fantasy Points']).reshape(1, -1)
		print("{} actual: {} || model {} prediction: {}".format(row['Player'], row['Fantasy Points'], i, model.predict(player)))
	i += 1
scores()