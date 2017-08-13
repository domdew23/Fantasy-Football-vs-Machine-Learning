import requests
from bs4 import BeautifulSoup

url = 'https://fantasydata.com/nfl-stats/player-details.aspx?playerid=7242-drew-brees'
source_code = requests.get(url)  # connects to url and stores info in variable source_code
plain_text = source_code.text  # takes text of request and stores it in plain_text (links, images etc.)
soup = BeautifulSoup(plain_text, 'html.parser')  # soup object is what you sort through

count = 0
for table in soup.find_all('table'):
	count += 1
	if count == 2:
		for tr in table.find_all('tr'):
			for td in tr.find_all('td'):
				print(td.text)