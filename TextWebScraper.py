from YAKE import YAKE
from tester import Evaluator
from csvReader import Reader
import sys, os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

sites = Reader("pinged_testing_dataset.csv").getSitesName()[0]
#print(sites)
annontedKeyewords = Reader("pinged_testing_dataset.csv").getSitesName()[1]
annontedKeyewords = annontedKeyewords[:5]
annontedKeyewords = list( map(lambda keys:keys.split(","),annontedKeyewords ) )
#print( annontedKeyewords )

getedKeyword = []
siteL=len(sites)

for index,site in enumerate(sites):
    try:
        path = "chromedriver.exe"
        chrome_option = Options()
        chrome_option.add_argument("--headless")
        chrome_option.add_argument('--no-sandbox')
        driver = webdriver.Chrome(path, options=chrome_option)
        driver.get(site)
        driver.page_source.encode("utf-8")
        text = driver.find_element_by_tag_name("body").get_attribute("innerText")
        print(text)
        extractor = YAKE(text).get_keyword()
        getedKeyword.append(extractor)
        print("extractor:",extractor)
        driver.close()
    except Exception as e:
        print( "site N:{} out of {} echeck:{}" .format( index, siteL,e.args) )
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    if index >5:
        break

""" 
evaluator = Evaluator(getedKeyword,annontedKeyewords)

print( "the accurecy is %d" % evaluator.precesion() )

print( "the rappel is %d" % evaluator.rappel() )

print( "the F1-score is %d" % evaluator.F1_mesure() )
        
"""    
    
    

