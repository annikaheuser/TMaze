import string
import pickle
import os

import wordfreq

class materials:
    def __init__(self,items,delimiter,WORK_PATH,lang,pickle_files=None,matching_distractors=False):
        '''
        items = list of strings, of the form:
        d;01;Die Mutter von Paula und die Schwester...
        d = Condition, 1 = Condition Index, Die Mutter... = Sentence
        In this case the delimiter is ";"
        Filler sentences should start with "filler"

        pickle_files = dict of the names of the files in which to store 
        the info files about the experimental materials
        Must include the following keys: "cond_dict" & "word_info"

        WORK_PATH = path to which all the file names will be appended
        '''
        self.cond_dict = {}
        self.word_info = {}
        self.num_item_pairs = {}
        #Can consider creating a specific materials folder
        self.WORK_PATH = WORK_PATH
        self.pickle_files = pickle_files
        self.lang_code = lang #might want to inherit this
        try:
            use_pkls,user_msg,pkl_paths = self.evaluate_pickle_situation()
        except LookupError as err: 
            print(err.args[0])
            raise LookupError
        if use_pkls:
            print(user_msg)
            with open(pkl_paths[0],"rb") as f:
                self.cond_dict = pickle.load(f)
            with open(pkl_paths[1],"rb") as f:
                self.word_info = pickle.load(f)
        else:
            print(user_msg)
            for item in items: 
                cond,ind,sent = item.split(delimiter)
                sent = sent[:-1] #get rid of \n
                try:
                    self.add_word_info(sent)
                except LookupError as err:
                    print(err.args[0])
                    print(f"Consider checking sentence {cond},{ind} for typos, not adding this sentence to experimental data.")
                    continue
                self.cond_dict.setdefault(cond,[]).append(sent) 
                self.num_item_pairs.setdefault(ind,{})[cond] = sent
            if pkl_paths:
                with open(pkl_paths[0], "wb") as f:
                    pickle.dump(self.cond_dict,f)
                with open(pkl_paths[1], "wb") as f:
                    pickle.dump(self.word_info,f)

    def add_word_info(self,sent):
        for word in sent.split():
            punct = False
            #formatted_word = word.lower()
            formatted_word = word
            if formatted_word[-1] in string.punctuation:
                formatted_word = formatted_word[:-1]
                punct = True
            if formatted_word not in self.word_info:
                freq = wordfreq.zipf_frequency(formatted_word,self.lang_code)
                if not freq:
                    if punct:
                        #Don't want to weird out the user by including punctuation
                        err_str = f"No data for \"{word[:-1]},\"is this a real word?"
                    else:
                        #Also don't want the user to think that the problem was 
                        #that the word was lowercase when it shouldn't have been
                        err_str = f"No data for \"{word}\", is this a real word?"
                    raise LookupError(err_str)
                self.word_info[formatted_word] = {"len":len(formatted_word), "freq": freq}

    def compile_exp_filler_sent_list(self):
        exp_sents = []
        filler_sents = []
        for cond in self.cond_dict:
            if cond != "filler":
                exp_sents+=self.cond_dict[cond]
            else:
                filler_sents+=self.cond_dict[cond]
        return {"exp": exp_sents, "filler": filler_sents}
    
    def check_for_pickle_file(self,key):
        #if key not in self.pickle_files:
            #raise LookupError(f"Missing key \"{key}\" in parameter \"pickle_files\"")
            #print(f"Missing key \"{key}\" in parameter \"pickle_files\"")
            #return False,None 
            #consider allowing a user to save just one
        if key in self.pickle_files:
            file_path = f'{self.WORK_PATH}/{self.pickle_files[key]}'
            if os.path.isfile(file_path):
                if os.stat(file_path).st_size != 0:
                    return True,file_path
        return False,file_path
    
    def evaluate_pickle_situation(self):
        if not self.pickle_files:
            user_msg = "Creating dictionaries from the list of sentences.\n"
            user_msg+="These won't be saved as pkl files because no file paths were provided. "
            return False,user_msg,None
        keys = ["cond_dict", "word_info"]
        exist = []
        file_paths = []
        for key in keys:
            exists_bool,file_path = self.check_for_pickle_file(key) #may result in a Lookup error
            exist.append(exists_bool) 
            file_paths.append(file_path)
        if all(exist):
            user_msg = "Both pickle files already exist, loading existing pickles as opposed to creating new ones."
            user_msg += "\nIf you want these files overwritten, please delete one or both of them from the directory."
            return True,user_msg,file_paths
        else:
            user_msg = "One or more of the pickle files are blank or do not exist, creating new dictionaries." 
            user_msg += "\nPotentially overwriting a file in the process."
            return False,user_msg,file_paths

        
                    
                




