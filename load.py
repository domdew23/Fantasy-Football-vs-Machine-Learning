import re
import json
import os
import requests
import numpy as np
import pandas as pd 

from bs4 import BeautifulSoup


def spider(url, file):
	source_code = requests.get(url)  # connects to url and stores info in variable source_code
	plain_text = source_code.text  # takes text of request and stores it in plain_text (links, images etc.)
	soup = BeautifulSoup(plain_text, 'html.parser')  # soup object is what you sort through
	# for each row in the table
	count = 0
	header = []
	data = []
	for table in soup.find_all('table'):
		count += 1
		if count == 2:
			for th in table.find_all('th'):
				head_text = th.text
				header.append(head_text)
			for tr in table.find_all('tr'):
				row = []
				for td in tr.find_all('td'):
					text = td.text
					row.append(text)
				if row:
					data.append(row)

	df = pd.DataFrame(data, columns=header)
	return df


def grab_data(url, output_file):
	if os.path.isfile(output_file):
		print("Url already crawled || File already exists")
	else:
		return spider(url, output_file)


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
				if text == '' or not td:
					continue
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
	tmp = pd.DataFrame(train_data, columns=header)
	return tmp


def merge(df):
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
	#df, predict_row = polish_df(df)
	return polish_df(df)


def polish_df(df):
	predict_row = df.loc[[0]]

	targets = df['Fantasy Points']
	targets.drop(targets.index[[-1]], inplace=True)

	df.drop(0, inplace=True)
	df.drop('Fantasy Points', axis=1, inplace=True)

	predict_row.drop('Fantasy Points', axis=1, inplace=True)
	df.reset_index(drop=True, inplace=True)

	df['Fantasy Points'] = targets
	#df.to_csv(file, index=False)
	return df, predict_row

def format_name(name):
	name = name.lower()
	name = name.split(' ')
	name.insert(1, '-')
	name = ''.join(name)
	return name


def save_players(url, file):
	if os.path.isfile(file):
		print("File already exists")
		return
	source_code = requests.get(url)  # connects to url and stores info in variable source_code
	plain_text = source_code.text  # takes text of request and stores it in plain_text (links, images etc.)
	soup = BeautifulSoup(plain_text, 'html.parser')  # soup object is what you sort through
	data = {}
	# for each row in the table
	for tr in soup.find_all('tr'):
		count = 0
		player = {}
		for td in tr.find_all('td'):
			if count == 1:
				player_id = td.text.strip()
				player['id'] = player_id
			elif count == 2:
				player_name = format_name(td.text.strip())
				player['name'] = player_name
				break
			count += 1
		if not data:
			if player:
				data['players'] = [player]
		else:
			data['players'].append(player)
	with open(file, 'w') as f:
		json.dump(data, f, indent=4)


def get_data(json_file):
	predictions = []
	with open(json_file, 'r') as f:
		data = json.load(f)
	for player in data['players']:
		player_id = player['id']
		name = player['name']
		link = 'https://fantasydata.com/nfl-stats/player-details.aspx?playerid={}-{}'.format(player_id, name)
		out_file = 'player_data/QB/{}.csv'.format(name)
		df = grab_data(link, out_file)
		final_df, predict_row = merge(df)
		predictions.append(predict_row)
		final_df.to_csv(out_file)

		print("{} Done.".format(name))
		print("Predictions values: {}".format(predict_row))


urls = ['http://www.footballoutsiders.com/stats/teamoff', 'http://www.footballoutsiders.com/stats/drivestatsoff']

get_data('players.json')
#link = 'https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=4&stype=0&sn=1&scope=0&w=0&ew=0&s=&t=0&p=2&st=FantasyPointsYahoo&d=1&ls=FantasyPointsYahoo&live=false&pid=true&minsnaps=4'
#save_players(link, 'players.json')

#merge(output_file)

