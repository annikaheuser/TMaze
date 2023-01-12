import pickle
import requests
import re
from bs4 import BeautifulSoup
import wordfreq
import pandas as pd
import os
import string
import spacy
#import spacy_transformers
import language_tool_python

class lang_spec:
    already_tested = ['en','de']
    wordfreq_supported = ['ar','bn','bs','bg','ca','zh','hr','cs','da','nl','en','fi','fr','de','el','he','hi','hu',
    'is','id','it','ja','ko','lv','lt','mk','ms','nb','fa','pl','pt','ro','ru','sl','sk','sr','es','sv','fil','ta',
    'tr','uk','ur','vi']
    spacy_supported = ['ca','zh','da','nl','en','fi','fr','de','el','it','ja','ko','lt','mk','nb','pl','pt','ro',
    'ru','es','sv']
    #TODO: add langtools supported list
    cap_pos = {'de': {'NE','NN'}, 'en': {'NNP','NNPS'}}

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    def __init__(self,lang_code,capitalize,WORK_PATH,spacy_pipeline=None,save=True,user_pkls={}):
        self.lang_code = lang_code
        self.lang = lang_code.split('-')[0] #luckily wordfreq chooses closest word code
        self.nlp_pipeline = None
        if capitalize and spacy_pipeline:
            self.nlp_pipeline = spacy.load(spacy_pipeline)
            if self.lang != spacy_pipeline.split("_")[0]:
                print("Please make sure that the language code you entered corresponds to the language code used by spacy.")
        self.capitalize = capitalize
        self.WORK_PATH = WORK_PATH #inherit this from tmaze???
        self.save = save
        self.user_pkls = user_pkls
        """ if lang_code in already_tested:
            self.existing_pkls = True """
        self.freq_bins = {}
        self.nonwords_set = set()
        
    def compile_freq_bins_and_nonwords_set(self):
        if not self.nonwords_set:
            self.compile_nonwords_set()
        if self.user_pkls is not None:
            if 'freq_bins' in self.user_pkls:
                #overrides use of existing pkls
                print("Loading freq_bins from the user-defined pkl.")
                with open(f'{self.WORK_PATH}/{self.user_pkls["freq_bins"]}','rb') as f:
                    self.freq_bins = pickle.load(f)
        if not self.freq_bins:
            if self.lang in self.already_tested:
                print(f"Loading freq_bins from pkl used by developers for {self.lang}.")
                #should be in the folder with this class
                with open(os.path.join(self.__location__, f'freq_bins_{self.lang}_ensemble.pkl'),'rb') as f:
                    self.freq_bins = pickle.load(f)
            elif self.lang in self.wordfreq_supported:
                if self.capitalize and self.nlp_pipeline:
                    #capitalizing will be based on spacy's nlp tagging
                    if self.lang not in self.cap_pos:
                        print("Please enter which part of speech (POS) tags should be capitalized.")
                        print("Make sure to use the same POS labels used by spacy.")
                        print("Enter them in the following format, with white space between each:" )
                        print("NNP NNPS")
                        self.add_user_tags()
                    #else:
                        #print("Capitalization of distractors not supported for {self.lang}.")
                self.make_freq_bins()
            else:
                print("Building of freq_bins not supported for {self.lang}.")
                print(f"Consider building this dictionary yourself, using the same structure as the frequency dictionaries included for English (en) or German (de).")

    def add_user_tags(self):
        raise NotImplementedError

    def make_freq_bins(self):
        if not self.nonwords_set: #likely redundant
            self.compile_nonwords_set()
        
        self.lang_tool = None
        if not self.nlp_pipeline:
            self.lang_tool = language_tool_python.LanguageTool(self.lang_code)
            capitalize_word = self.capitalize_word_langtool
        else:
            capitalize_word = self.capitalize_word_spacy
        #default is wordlist = 'best', uses 'large' if it's available and 'small' otherwise
        for word, freq in wordfreq.get_frequency_dict(self.lang).items():
            if not self.check_nonword(word):
                zipf = wordfreq.zipf_frequency(word, self.lang)
                self.freq_bins[zipf] = self.freq_bins.get(zipf, []) + [capitalize_word(word)]
        if self.lang_tool:
            self.lang_tool.close()

    def capitalize_word_spacy(self,word):
        tag = self.nlp_pipeline(word)[0].tag_
        if self.lang == "en":
            capitalized_word = self.capitalization_en_spec(word,tag=tag)
            if capitalized_word:
                return capitalized_word
        #assuming the language and the POS tags that should be capitalized have been added to the pos_cap dictionary
        if tag in self.cap_pos[self.lang]:
            return word[0].upper()+word[1:]
        return word

    def check_replacements(self,word,match):
        if match:
            to_check = match[0].replacements[:3] #no need to check beyond this
            for replacement in to_check:
                if replacement.lower() == word:
                    return replacement
        return word

    def capitalize_word_langtool(self,word):
        match = self.lang_tool.check(word)
        if self.lang == "en":
            capitalized_word = self.capitalization_en_spec(word,matches=match)
            if capitalized_word:
                return capitalized_word
        return self.check_replacements(word,match)
        

    def check_nonword(self,word):
        if word in self.nonwords_set:
            return True
        if any(char.isdigit() for char in word):
            return True
        if "." in word or "," in word:
            return True
        if self.lang == "en":
            return self.check_nonwords_en_spec(word)
        return False

    def check_nonwords_en_spec(self,word):
        vowels = {"a","e","i","o","u","y"}
        if not any(vowel in word for vowel in vowels):
            return True
        return False

    def handle_apostrophe_cases(self,word):
        apos_split = word.split("'") #this should always be length 2
        prefix = apos_split[0]
        cap_word = self.capitalize_word_langtool(prefix)
        cap_word+="'"
        suffix = apos_split[1]
        if cap_word == "o'": #thanks Irish for this special case
            cap_word = "O'"
            cap_word+=self.capitalize_word_langtool(suffix)
        else:
            cap_word+=suffix
        return cap_word

    def capitalization_en_spec(self,word,matches=None,tag=None):
         #all are also actual words but much less common than the names
        names_to_capitalize = {"john","mike","lee","max","harry","sally","tony","jimmy","roger","josh","johnny","maria",
                "rick","ian","graham","ted","cooper","khan","terry","nelson","mama","bobby","batman","sid","james",
                "welsh","rogers","yang","molly","belle","sims","parsons","lea","sheila","raj","oscars","modi", "potter",
                "trump's","incheon","assange","wales","yorker","bitcoin","kyrie","china"}
        only_nnp_in_context = {"financial","national","code","league","federal","trust","bank","east","west",
                "north","south","park","middle","club","military","department","board","father","mother","street","red",
                "international","house", "star","army","truth","wall","minister","box","association","sun","gun","fit",
                "master","gay","united","professor","democratic","series", "house","white","god","god's","lighten","gulf",
                "tokens"}
                #"God" might offend people? idk man
        if word in only_nnp_in_context:
                 #checked until 4.9 and thought of a few others while testing the sm pipeline
                return word
        if word in names_to_capitalize:
            #both the sm and tf pipelines mess up on some of these
            return word[0].upper()+word[1:]
        if not self.nlp_pipeline:
            #for lang_tool method
            if word == "i": #not caught
                return "I"
            if "'" in word:
                return self.handle_apostrophe_cases(word)
        else:
            if tag == "PRP":
                apos_split = word.split("'")
                if apos_split[0] == 'i':
                    if len(word) > 1:
                        return "I"+word[1:]
                    else:
                        return "I"

    def capitalization_ensemble(self):
        pkl_files = ["spacy_sm.pkl","spacy_trf.pkl","langtool.pkl"]
        file_paths = []
        fb_dict_dict= {}
        #check that we have all the pkl files we need
        for file_name in pkl_files:
            file_path = os.path.join(self.__location__,f"freq_bins_{self.lang}_{file_name}")
            if os.path.isfile(file_path):
                file_paths.append(file_path)
            else:
                #decide whether to produce the missing one individually or just use them all at the same time
                print("Ensemble capitalization currently requires the freq_bins dictionary produced by each model.")
                raise NotImplementedError
        key_list = []    
        for file_path,pkl_name in zip(file_paths,pkl_files):
            key = pkl_name.split(".")[0]
            key_list.append(key)
            with open(file_path,"rb") as handle:
               fb_dict_dict[key] = pickle.load(handle)
        key1,key2,key3 = key_list
        fb_dict1,fb_dict2,fb_dict3 = fb_dict_dict[key1],fb_dict_dict[key2],fb_dict_dict[key3]
        for freq in fb_dict1:
            for i in range(len(fb_dict1[freq])):
                word_forms = [fb_dict1[freq][i],fb_dict2[freq][i],fb_dict3[freq][i]]
                mode = max(set(word_forms), key = word_forms.count)
                self.freq_bins[freq] = self.freq_bins.get(freq, []) + [mode]

    def compile_nonwords_set(self):
        if 'nonwords_set' in self.user_pkls:
            print("Loading nonwords_set from the user-defined pkl.")
            with open(f'{self.WORK_PATH}/{self.user_pkls["nonwords_set"]}','rb') as f:
                self.nonwords_set = pickle.load(f)
        if not self.nonwords_set:
            if self.lang in self.already_tested:
                print(f"Loading nonwords_set from pkl used by developers for {self.lang}.")
                #should be in the folder with this class
                with open(os.path.join(self.__location__, f'nonwords_{self.lang}.pkl'),'rb') as f:
                    self.nonwords_set = pickle.load(f)
    
    def nonwords_from_txt(self,file_path):
        with open(file_path) as f: #combined fillers and test sentences 
            words = f.readlines()
        for word in words:
            self.nonwords_set.add(word[:-1]) #to remove the new line character

    def compile_nonwords_en(self):
        #Script for creating a nonwords set in English
        #Provided as an example and for transparency

        self.nonwords_set = {"j00","lmao","exp","mph","calif","usda","td","isis","dont","gofundme",
        "hitler","http","lgbt", "ofthe", "ncaa","https","mum","feb","mar","apr","jun","jul","aug","sept",
        "oct","nov","dec","fuckin","il","nazi","yea","huh","cock","ugh","nasa","fifa","mps","ba","sri",
        "prof","ios","bmw","l.a","opt","gosh","hon","cuz","ahh","ot","dong","bf","soo","og","tis","pres",
        "duh","shes","hes","youre","im","ive","youve","id","youd","hed","weve","dnc","umm","um","tor","eps",
        "sup","p.s","dt","sgt","phi","org","mmm","dat","ptsd","casa","yrs","cps","ooh","heck","crap","linkedin",
        "hell","junkie","memorise","programme", "lmfao","didnt","pics","isnt","cutie","meme","ain't","stoned",
        "auto","esque","thru","whats","gonna","thats","haha","asap","dhabi","doesnt","centre","mins","yeah",
        "copulation", "sex", "didnt","wouldnt","programmes","maximise","dude"}
        #figure out how to get it to use a file in the same folder 
        self.nonwords_from_txt(os.path.join(self.__location__, 'exclude_en.txt'))
        
        #collection of words we don't want excluded through the wikipedia scrape
        #consider throwing out every two-letter word that isn't one of these
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
        actual_words = {"a","i"} | common_2letter | common_3letter | other_noticed

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

    def delete_nonwords_after(self,nonwords):
        self.nonwords_set|=set([nonword.lower() for nonword in nonwords])
        for freq,word_list in self.freq_bins.items():
            new_word_list = [word for word in word_list if word not in nonwords]
            self.freq_bins[freq] = new_word_list

    def switch_word_cap(self,words_to_switch):
        switched_words = [word[0].upper()+word[1:] if word[0].islower() else word[0].lower()+word[1:] for word in words_to_switch]
        for freq,word_list in self.freq_bins.items():
            ind = 0
            new_word_list = word_list[:]
            for word in word_list:
                if word in words_to_switch:
                    word_ind = words_to_switch.index(word)
                    new_word_list = new_word_list[:ind] + [switched_words[word_ind]] + new_word_list[ind+1:]
                ind+=1
            self.freq_bins[freq] = new_word_list

            



            









         
    

            
                
                
            
        
                



        


