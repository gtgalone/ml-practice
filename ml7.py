import requests
import bs4 from BeautifulSoup

session = requests.session()

url = "http://google.com/login"
data = {
  "return_url": "10",
  "m_id": "20",
  "m_password": "30"
}

response = session.post(url, data=data)
response.raise_for_status()

url = "http://google.com/mypage"
response = session.get(url)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")
text = soup.select_one(".mileage_section2 span").get_text()

print("마일리지:", text)

# session.post()
# session.put()
# session.delete()