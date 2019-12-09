import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd

##forbid the pic
options = webdriver.ChromeOptions()
prefs = {'profile.default_content_setting_values': { 'images': 2}}
options.add_experimental_option('prefs',prefs)
driver = webdriver.Chrome(chrome_options=options)

##get to main page and get target list
driver.get('https://store.steampowered.com/sale/2016_top_sellers/')
targetlinks = driver.find_elements(By.XPATH,'//div[@data-ds-appid]//a')##元素列表
link = []
for i in targetlinks:
    targetlink = i.get_attribute('href')
    link.append(targetlink)

##define the detail space
detailtextlistsum = []
detailname_list = []
tagtextlistsum = []

##get into target page and try to seek web element
for x in link:
    driver.get(x)
    try:
        detailinfo = driver.find_element(By.CLASS_NAME, 'apphub_AppName')
        ##detail part
        detail = driver.find_elements_by_xpath("//div[@class='game_area_sys_req sysreq_content active' and @data-os='win']//li")
        tag = driver.find_elements_by_xpath("//*[@id='game_highlights']/div[1]/div/div[4]/div/div[2]/a")


    except:
        print('error')
    else:
         tagtext_list = []
         for t in tag[0:4]:
            tagtext_list.append(t.get_attribute('textContent'))
         detailtext_list = []
         for y in detail:
            detailtext_list.append(y.get_attribute('textContent'))
         detailtextlistsum.append(detailtext_list)
         detailname_list.append(detailinfo.text)
         tagtextlistsum.append(tagtext_list)

print(tagtextlistsum)




tag_form = pd.DataFrame(index=detailname_list,data=tagtextlistsum)
detailtext_form = pd.DataFrame(index=detailname_list,data=detailtextlistsum)
# write in csv
detailtext_form.to_csv('C:/python/testdata/requirement.csv',encoding='utf-8-sig')
tag_form.to_csv('C:/python/testdata/tag.csv',encoding='utf-8-sig')
print(detailtext_form)
print(tag_form)


//*[@id="tag_browse_games_ctn"]/div/div[1]/div[1]/div[1]/div[1]/a
//*[@id="tag_browse_games_ctn"]/div/div[1]/div[1]/div[2]/div[1]/a
//*[@id="tag_browse_games_ctn"]/div/div[1]/div[1]/div[9]/div[1]/a


