import urllib.request

url = "https://api.aoikujira.com/ip/ini"

# download
mem = urllib.request.urlopen(url).read()

print(mem.decode("utf-8"))