from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR, SVR, NuSVR
import numpy as np

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



def make_models(X_train, y_train):

	ridge = Ridge(alpha=1000, max_iter=1000000, random_state=0).fit(X_train, y_train) # 0 
	lasso = Lasso(alpha=100, max_iter=1000000, random_state=0).fit(X_train, y_train) # 1

	svr = SVR(kernel='linear', C=.01, epsilon=0.1, gamma=7, max_iter=10000, verbose=True).fit(X_train, y_train) # 3
	lin_svr = LinearSVR(C=.01, epsilon=0.0001, max_iter=10000, verbose=0, random_state=0).fit(X_train, y_train) # 4
	nu_svr = NuSVR(C=.1, gamma=7, verbose=True).fit(X_train, y_train) # 9

	return ridge, lasso, svr, lin_svr, nu_svr



def graph_ridge(ridge, lasso, svr, lin_svr):
	plt.plot(ridge.coef_, 's', label='Ridge')
	plt.plot(lasso.coef_, '^', label='Lasso')
	#plt.plot(svr.coef_, 'v', label='SVR')
	plt.plot(lin_svr.coef_, 'o', label='Linear SVR')

	plt.xlabel("Coefficient index")
	plt.ylabel("Coefficent magnitude")
	plt.hlines(0, 0, len(svr.coef_))
	plt.legend()
	plt.show()


def scores():
	print_scores(ridge,'Ridge')
	print_scores(lasso,'Lasso')
	print_scores(svr,'SVR')
	print_scores(lin_svr,'LinSVR')
	print_scores(nu_svr,'NuSVR')



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
graph_ridge(ridge, lasso, svr, lin_svr)



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