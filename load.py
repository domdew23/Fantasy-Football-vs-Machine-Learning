import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR, NuSVR
import re

def print_scores(model, model_type=''):
	train_score = model.score(X_train, y_train)
	test_score = model.score(X_test, y_test)
	print(model_type + " Training score: {:.2f}".format(train_score))
	print(model_type + " Testing score: {:.2f}".format(test_score))
	try:
		print(model_type + " Number of features used: {}".format(np.sum(model.coef_ != 0)))
	except ValueError:
		print("All features used")
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


def grab_data(url, output_file):
	if os.path.isfile(output_file):
		print("Url already crawled || File already exists")
	else:
		spider(url, output_file)


def scores():
	print_scores(ridge,'Ridge')
	print_scores(lasso,'Lasso')
	print_scores(svr,'SVR')
	print_scores(lin_svr,'LinSVR')
	print_scores(nu_svr,'NuSVR')


def make_models(X_train, y_train):

	ridge = Ridge(alpha=1000, max_iter=1000000, random_state=0).fit(X_train, y_train) # 0 
	lasso = Lasso(alpha=100, max_iter=1000000, random_state=0).fit(X_train, y_train) # 1

	svr = SVR(kernel='linear', C=.01, epsilon=0.1, gamma=7, max_iter=10000, verbose=True).fit(X_train, y_train) # 3
	lin_svr = LinearSVR(C=.01, epsilon=0.0001, max_iter=10000, verbose=0, random_state=0).fit(X_train, y_train) # 4
	nu_svr = NuSVR(C=.1, gamma=7, verbose=True).fit(X_train, y_train) # 9

	return ridge, lasso, svr, lin_svr, nu_svr


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

def crawl(team, year, url, header, get_header=True):
	source_code = requests.get(url)  # connects to url and stores info in variable source_code
	plain_text = source_code.text  # takes text of request and stores it in plain_text (links, images etc.)
	soup = BeautifulSoup(plain_text, 'html.parser')  # soup object is what you sort through
	# for each row in the table
	rows = [year]
	row_num = 0
	row_found = False
	for tr in soup.find_all('tr'):
		team_found = False
		# for each entry in a row
		if not row_found:
			for td in tr.find_all('td'):
				# print each entry in that row
				text = td.text.strip()
				if row_num == 0 and get_header:
					if text == 'NON-ADJUSTED':
						non_adj = ['Non-adj total', 'Non-adj Pass', 'Non-adj Rush']
						header.extend(non_adj)
					else:
						if text != '' and text.lower() != 'team':
							header.append(text.title())
				if text == team:
					team_found = True
					continue
				if team_found:
					text = filter(text)
					rows.append(text)
		if team_found:
			row_found = True
		row_num += 1
	return header, rows


def filter(text):
	text = re.sub("[\(\[].*?[\)\]]", "", text).strip()
	if ":" in text:
		text = text.replace(":", ".")
		tmp = text.split('.')
		val = (int(tmp[0]) * 60) + int(tmp[1])
		text = float(val)
	elif "%" in text:
		text = text.replace("%", "")
		text = float(text) / 100
	text = float(text)
	return round(text, 3)


def fill(df, url):
	train_data = []
	predict_data = []
	header = ['Season']
	i = 0
	get_header = True
	for index, row in df.iterrows():
		if i != 0:
			get_header = False
		tmp_url = '{}{}'.format(url, row['Season'])
		header, year = crawl(row['Team'], row['Season'], tmp_url, header, get_header)
		train_data.append(year)
		i = 1
	df = pd.DataFrame(train_data, columns=header)
	return df


def merge(file):
	df = pd.read_csv(file)
	seasons = df['Season']
	data_frames = []
	for url in urls:
		d = fill(df, url)
		data_frames.append(d)

	df1 = pd.merge(data_frames[0], data_frames[1], on='Season')

	tmp = ['Season', 'Fantasy Points']
	tmp_df = pd.DataFrame(df[tmp])

	df = df.drop('Fantasy Points', axis=1)
	df = pd.merge(df, df1, on='Season')
	df = pd.merge(df, tmp_df, on='Season')
	df.to_csv('drew_brees_final.csv')


def load_player(file):
	df = pd.read_csv(file)

	name = str(df['Player'][0])
	predict = df.loc[[0]]

	targets = df['Fantasy Points']
	targets.drop(targets.index[[-1]], inplace=True)

	df.drop(0, inplace=True)
	df.drop('Fantasy Points', axis=1, inplace=True)
	predict.drop('Fantasy Points', axis=1, inplace=True)
	df.reset_index(drop=True, inplace=True)
	cols = [1,2,3]

	df.drop(df.columns[cols], axis=1, inplace=True)
	predict.drop(predict.columns[cols], axis=1, inplace=True)
	#df['Fantasy Points'] = targets
	return df, targets, name, predict


urls = ['http://www.footballoutsiders.com/stats/teamoff', 'http://www.footballoutsiders.com/stats/drivestatsoff']
#url = 'https://fantasydata.com/nfl-stats/player-details.aspx?playerid=7242-drew-brees-new-orleans-saints'
output_file = 'player_data/QB/drew_brees.csv'
#grab_data(url, output_file)

#merge(output_file)
X, y, name, predict = load_player('drew_brees_final.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


ridge, lasso, svr, lin_svr, nu_svr = make_models(X_train, y_train)
models = [ridge, lasso, svr, lin_svr, nu_svr]

tmp = ['%.2f' % x for x in y_test]

i = 0
for m in models:
	diffs = []
	predictions = m.predict(X_test)
	predictions = ['%.2f' % x for x in predictions]
	for pred, actual in zip(predictions, tmp):
		diff = float(pred) - float(actual)
		diffs.append(abs(diff))
	mean = sum(diffs) / len(tmp)
	print("{} prediction by model {}: {} || mean difference: {} \n\t\t   season actual: {}".format(name, i, predictions, mean, tmp))
	i += 1
 
scores()
'''
X_train, y_train, train_names = load_file('data.csv')
X_test, y_test, test_names = load_file('test_data_2014.csv')

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
'''