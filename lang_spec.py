import pickle
import requests
import re
from bs4 import BeautifulSoup
import wordfreq
import pandas as pd
import os
import string

class lang_spec:
    already_tested = ['en','de']
    wordfreq_supported = ['ar','bn','bs','bg','ca','zh','hr','cs','da','nl','en','fi','fr','de','el','he','hi','hu',
    'is','id','it','ja','ko','lv','lt','mk','ms','nb','fa','pl','pt','ro','ru','sl','sk','sr','es','sv','fil','ta',
    'tr','uk','ur','vi']
    spacy_supported = ['ca','zh','da','nl','en','fi','fr','de','el','it','ja','ko','lt','mk','nb','pl','pt','ro',
    'ru','es','sv']
    cap_pos = {'de': {'NE','NN'}, 'en': {'NNP','NNPS'}}

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    def __init__(self,lang_code,capitalize,WORK_PATH,save=True,user_pkls=None):
        self.lang_code = lang_code
        self.capitalize = capitalize
        self.WORK_PATH = WORK_PATH #inherit this from tmaze???
        self.save = save
        self.user_pkls = user_pkls
        """ if lang_code in already_tested:
            self.existing_pkls = True """
        self.freq_dict = None
        self.nonwords_set = None
        
    def check_freq_dict_options(self):
        if self.user_pkls is not None:
            if 'freq_dict' in self.user_pkls:
                #overrides use of existing pkls
                print("Loading freq_dict from the user-defined pkl.")
                with open(f'{self.WORK_PATH}/{self.user_pkls["freq_dict"]}','rb') as f:
                    self.freq_dict = pickle.load(f)
        if not self.freq_dict:
            if self.lang_code in self.already_tested:
                print(f"Loading freq_dict from pkl used by developers for {self.lang_code}.")
                #should be in the folder with this class
                with open(os.path.join(self.f__location__, f'FreqToDict_{self.lang_code}'),'rb') as f:
                    self.freq_dict = pickle.load(f)
            elif self.lang_code in self.wordfreq_supported:
                if self.lang_code in self.spacy_supported:
                    print("Building of freq_dict with capitalization supported for {self.lang_code}.")
                    if self.lang_code not in self.cap_pos:
                        print("Please enter which part of speech (POS) tags should be capitalized.")
                        print("Make sure to use the same POS labels used by spacy.")
                        print("Enter them in the following format, with white space between each:" )
                        print("NNP NNPS")
                        self.add_user_tags()
                else:
                    print("Capitalization of distractors not supported for {self.lang_code}.")
            else:
                print("Building of freq_dict not supported for {self.lang_code}.")
                print(f"Consider building this dictionary yourself, using the same structure as the frequency dictionaries included for English (en) or German (de).")

    def add_user_tags(self):
        raise NotImplementedError

    def make_freq_dict(self):
        pass

    def compile_nonwords_set(self):
        if 'nonwords_set' in self.user_pkls:
            print("Loading nonwords_set from the user-defined pkl.")
            with open(f'{self.WORK_PATH}/{self.user_pkls["nonwords_set"]}','rb') as f:
                self.nonwords_set = pickle.load(f)
        if not self.nonwords_set:
            if self.lang_code in self.already_tested:
                print(f"Loading nonwords_set from pkl used by developers for {self.lang_code}.")
                #should be in the folder with this class
                with open(os.path.join(self.f__location__, f'nonwords_{self.lang_code}.pkl'),'rb') as f:
                    self.nonwords_set = pickle.load(f)
    
    def nonwords_from_txt(self,file_path):
        with open(file_path) as f: #combined fillers and test sentences 
            words = f.readlines()
        for word in words:
            self.nonwords_set.add(word[:-1]) #to remove the new line character

    def compile_nonwords_en(self):
        #Script for creating a nonwords set in English
        #Provided as an example and for transparency

        self.nonwords_set = {"j00","lmao","u.s","exp","mph","calif","usda","td","i.e","isis",
        "dont","gofundme","hitler","http","lgbt", "ofthe", "ncaa"}
        #figure out how to get it to use a file in the same folder 
        self.nonwords_from_txt(os.path.join(self.__location__, 'exclude_en.txt'))
        
        #collection of words we don't want excluded through the wikipedia scrape
        common_2letter = {'of','to','in','it','is','be','as','at','so','we','he','by','or','on','do','if','me','my','up','an',
        'go','no','us','am'}
        common_3letter = {'the','and','for','are','but','not','you','all','any','can','had','her','was','one','our','out',
        'day','get','has','him','his','how','man','new','now','old','see','two','way','who','boy','did','its','let','put',
        'say','she','too','use'}
        other_noticed = {"hi","abs","ace","act", "ad","add","ads","age","ago","aid","aim","air","ale","ant","ape","app","arc",
        "ark","arm","art","ash","ask","ate","awe","axe","bad","bag","ban","bar","bat","bay","bed","bee","beg","ben","bet",
        "bib","bid","big","bin","bit","bog","bot","bow","box","bra","bud","bug","bun","bus","buy","bye","cab","cam","cap",
        "car","cat","caw","cod","cog","con","coo","cop","cot","coy","dam","dan","die","dig","dim","din","dip","dog","dry",
        "ego","emo","fab","fad","fan","far","fat","few","fib","fin","fig","fit","gas","gel","get","gig","gin","gum","gun",
        "gut","guy","ham","hat","hit","hog","hot","hut","ire","jam","kin","lad","lay","led","lie","low","mad","map","max",
        "mix","net","nun","oak","odd","off","pad","pam","pan","pay","peg","pen","pet","pin","pop","ram","rap","red","rim",
        "rip","rod","rub","rye","sad","sap","sin","sit","sly","son","spa","sun","ted","tea","tim","tin","war","wax","win"}
        actual_words = {"a"} | common_2letter | common_3letter | other_noticed

        def check_for_word(alleged_acr):
            if len(alleged_acr) >= 4: #will also exclude some actual words that aren't very frequent like capes
                if wordfreq.zipf_frequency(alleged_acr,"en"):
                    return True
            elif alleged_acr in actual_words:
                return True
            return False
        
        def check_for_matches(regex_str,str_to_search,new_acrs):
            matches = re.search(regex_str,str_to_search)
            if matches:
                for m in matches.groups():
                    acr = m.lower()
                    if not check_for_word(acr):
                        self.nonwords_set.add(acr)

        #scrape acronyms from wikipeia
        base_url = "https://en.wikipedia.org/"
        url = "https://en.wikipedia.org/wiki/Lists_of_acronyms"
        acr_regexes = ['(^[a-zA-Z0-9]+)((?= +–)|(?=$))','(^[a-zA-Z0-9]+)(?: or )([a-zA-Z0-9]+)((?= +–)|(?=$))']

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find("ul").find_all("li")[1:]:
            suff = link.find('a', href=True)["href"]
            out_url = base_url+suff
            resp = requests.get(out_url)
            acr_soup = BeautifulSoup(resp.text, 'html.parser')
            for unord_list in acr_soup.find_all("ul")[2:-17]:
                str_unord_list = unord_list.text
                new_acrs = set()
                for line in str_unord_list.split("\n"):
                    for acr_re in acr_regexes:
                        check_for_matches(acr_re,line,new_acrs)

    def compile_nonwords_de(self):
        #Script for creating a nonwords set in English
        #Provided as an example and for transparency
        ''' Shouldn't need the ones that start with 0 after change to main script
        self.nonwords_set = {"00.00","00,0","00000","'0,00'", "000.000", "000", "00", "0000","00ern", "afd", "fc", "cdu", "gmbh", "vllt", "bmfsfj", "spd", "http",
            "https", "00,00", "ii", "iii", "000,0", "km", "high", "night", "king", "time", "iq", "nsdap", "fpö", "evtl", "pkw", "00h", "0000ern", 
            "0000p", "heer", "jobst", "nice", "kpdsu", "dsgvo", "bgbl", "00ers", "00fps", "00min", "000ml", "fcb", "rbtv",  "fdp", "0000er", "00er",
            "cm", "00,00", "csu"}
        '''
        self.nonwords_set = {"afd", "fc", "cdu", "gmbh", "vllt", "bmfsfj", "spd", "http", "https", "ii", "iii", "km", "high", "night",
            "king", "time", "iq", "nsdap", "fpö", "evtl", "pkw", "heer", "jobst", "nice", "kpdsu", "dsgvo", "bgbl", "fcb", "rbtv",  "fdp",
            "cm", "csu"}
        url = "https://en.wikipedia.org/wiki/List_of_German_abbreviations"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables=soup.find_all('table',{'class':"wikitable"})[1:] #first table just has the sources
        for table,i in zip(tables,range(len(tables))):
            if not i:
                df = pd.DataFrame(pd.read_html(str(table))[0])
            elif i!=2:
                next_df = pd.DataFrame(pd.read_html(str(table))[0])
                df = df.append(next_df)
        actual_words = {"app","art","dir","max","schweiz",}
        for abr in df["Abbreviation"][5:]: #don't need to add random single characters or "am"
            abr = str(abr).lower().translate(str.maketrans('', '', string.punctuation))
            if " " not in abr and abr not in actual_words:
                self.nonwords_set.add(abr)

            









         
    

            
                
                
            
        
                



        


