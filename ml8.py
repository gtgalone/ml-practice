from selenium import webdriver

browser = webdriver.PhantomJS()

browser.implicitly_wait(3)

browser.get("https://nid.naver.com/nidlogin.login")

element_id = browser.find_element_by_id("id")
element_id.clear()
element_id.send_keys("10")
element_pw = browser.find_element_by_id("pw")
element_pw.clear()
element_pw.send_keys("10")

button = browser.find_element_by_css_selector("input.btn_global[type=submit]")
button.submit()

browser.get("https://mail.naver.com")

browser.save_screenshot("Website.png")
titles = browser.find_elements_by_css_selector("strong.mail_title")
for title in titles:
  print("-", title.text)


browser.quit()