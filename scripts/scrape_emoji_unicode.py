import pickle as pickle
from bs4 import BeautifulSoup
from urllib2 import urlopen, HTTPError

URL = 'https://apps.timwhitlock.info/emoji/tables/unicode'
soup = BeautifulSoup(urlopen(URL), 'lxml')

table_bytes = soup.findAll('td', {'class': 'code'})
emojis = set()
for td in table_bytes:
    if 'U+' in td.text:
        continue
    emojis.add(td.text)

emojis = list(emojis)
emojis = map(lambda x: x.decode('string_escape').decode('utf-8'), emojis)
with open('emoji_unicode.pkl', 'w') as f:
    pickle.dump(emojis, f)
