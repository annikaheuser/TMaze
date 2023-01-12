import pickle
import string
from scipy.stats import norm
import spacy
import numpy as np
import pandas as pd
import time


class tmaze: #TODO: think of better name
    #TODO: utilize WORK_PATH in init
    def __init__(self,scorer,spacy_pipeline,WORK_PATH,pickle_dict):
        self.WORK_PATH = WORK_PATH
        self.scorer = scorer
        self.nlp = spacy.load(spacy_pipeline)
        self.lang = spacy_pipeline.split("_")[0] #still need this for get_tag
        self.pickle_dict = pickle_dict
        if "freq_dict" in pickle_dict:
            with open(pickle_dict["freq_dict"], "rb") as f:
                self.freq_dict = pickle.load(f)
        else: #TODO
            print("Sorry, I haven't added the code yet to handle not already having a pickled freq_dict.")
        if "word_info" in pickle_dict:
            with open(pickle_dict["word_info"], "rb") as f:
                self.word_info = pickle.load(f)
        else: #TODO
            print("Sorry, I haven't added the code yet to handle not already having a pickled word_info.")
        if "nonwords_set" in pickle_dict:
            with open(pickle_dict["nonwords_set"], "rb") as f:
                self.nonwords_set = pickle.load(f)
        else: #TODO
            print("Sorry, I haven't added the code yet to handle not already having a pickled nonwords_set.")
        if "dists_dict" in pickle_dict:
            with open(pickle_dict["dists_dict"], "rb") as f:
                self.dists_dict = pickle.load(f)
        else: #TODO
            print("Please add a pkl file path to keep track of potential distractors.")



    def create_sent(self,ind,distractor,sent_list,just_preceding=False):
        '''ind = int, index where the distractor should be inserted
        distractor = string, to be inserted in the sentence
        sent_list = list of strings in correct order of sentence
        just_preceding = bool, whether to only include the words of the sentence up to the distractor
        Returns string that can be scored to determine how "surprising" the distractor is'''
        #check for whether the distractor should have punctuaction
        if sent_list[ind][-1] in string.punctuation:
            distractor+=sent_list[ind][-1]
        if not ind: #case 1: distractor is the first word
            sent = distractor
        else: 
            sent = ' '.join(sent_list[:ind])
            sent+=' '+distractor
            #don't need to do more in case 2, that distractor is the last word
        if ind+1 != len(sent_list) and not just_preceding: #general case
            sent+=' '
            sent+=' '.join(sent_list[ind+1: ])
        return sent
    


    def init_best_list(self,n):
        '''Initialize a list of tuples that will contain the n best distractors and their PLL scores
        n = int, for the number of top distractors that we want to keep track of
        '''
        init_tup = (0,"")
        return [init_tup]*n

    def any_tagged(self,distractors):
        tagged = False
        all_tagged = True
        some_tagged = False
        for dist in distractors:
            if not type(dist) == tuple:
                all_tagged = False
            else:
                some_tagged = True
        if all_tagged:
            tagged = True
        return tagged,some_tagged
    #TODO: write a function that only keeps track of the best distractor so far
    #Would probably be faster and is the main use-case
    def eval_best(self,ind,distractors,sent,best_so_far,just_preceding=False):
        evaluated = set()
        dist_sents = []
        for dist in distractors:
            if type(dist) == tuple:
                dist_sents+=[self.create_sent(ind,dist[0],sent,just_preceding)]
            else:
                dist_sents+=[self.create_sent(ind,dist,sent,just_preceding)]
        
        sent_plls = self.scorer.score_sentences(dist_sents)
        for i in range(len(distractors)):
            tagged = False
            dist_pll = sent_plls[i]
            if type(distractors[i]) == tuple:
                dist,tag = distractors[i]
                tagged = True
            else: #can't be a set because we need the indices to correspond to their plls
                dist = distractors[i]
            if dist_pll < best_so_far[0][0]: #replace a distractor in best_so_far
                if tagged:
                    best_so_far[0] = (dist_pll,dist,tag)
                else:
                    best_so_far[0] = (dist_pll, dist)
                best_so_far = sorted(best_so_far,reverse=True) #in order of worst to best
                #PLL scores are negative and the lower the PLL the better the distractor 
                #so highest to lowest (i.e. reversed) = worst to best ordering
            evaluated.add(dist)
        return best_so_far,evaluated

    def eval_over_two_sents(self,ind,distractors,sents,best_so_far,just_preceding=True):
        evaluated = set()
        #can definitely be smarter about this - make a functon after getting everything to work
        dist_sents1 = []
        dist_sents2 = []
        sent1,sent2 = sents
        for dist in distractors:
            dist_sents1+=[self.create_sent(ind,dist,sent1,just_preceding)]
            dist_sents2+=[self.create_sent(ind,dist,sent2,just_preceding)]
        #dist_sents get proper punctuation
        sent_plls1 = self.scorer.score_sentences(dist_sents1)
        sent_plls2 = self.scorer.score_sentences(dist_sents2)
        for i in range(len(distractors)):
            dist_pll = (sent_plls1[i]+sent_plls2[i])/2
            dist =  distractors[i]
            if dist_pll < best_so_far[0][0]: #replace a distractor in best_so_far
                best_so_far[0] = (dist_pll, dist)
                best_so_far = sorted(best_so_far,reverse=True)
            evaluated.add(dist)
        return best_so_far,evaluated

    #TODO: add mean and std as parameters for init
    def get_len_range(self,word_len,cdf_range=0.2): #using the power of statistics!
        ''' Determines the outer bounds of the length range 
        matching the word we're replacing with a distractor
        word_len = int, length of the word we're replacing with a distractor
        cdf_range = float, what percentage under the curve the length range should span
        '''
        #German
        #mean = 7.94 
        #std = 2.24
        #min_word_len = 2
        #English
        mean = 7.26
        std = 2.28
        min_word_len = 1 #unnecessary
        #TODO: add these to lang_spec class
        ppf = norm(loc=mean,scale=std).cdf(word_len)
        cdf_incr = cdf_range/2
        ppf_range = [ppf-cdf_incr, ppf+cdf_incr]
        dist_to_lower = 0
        dist_to_upper = 0
        #check if hitting lower limit
        if ppf_range[0] < 0.01:
            ppf_range[0] = 0.01
        #check if hitting upper limit
        if ppf_range[1] > 0.99: 
            ppf_range[1] = 0.99
        #lower_zscore = int(norm(loc=mean,scale=std).ppf(ppf_range[0]))
        #return [lower_zscore if lower_zscore >= min_word_len else min_word_len, int(norm(loc=mean,scale=std).ppf(ppf_range[1]))]
        return [int(norm(loc=mean,scale=std).ppf(ppf_range[0])),int(norm(loc=mean,scale=std).ppf(ppf_range[1]))]

    def get_tag(self,dist,ind,sent,just_preceding):
        '''
        Determine the passed distractor's STTS part of speech (POS) tag,
        given the context in which the participant will see the distractor 
        (or the entire sentence, depending on just_preceding)

        Parameters
        dist = string, the distractor we want to POS tag
        just_preceding = boolean, whether the context in which to determine the POS tag
        should only include the part of the sentence preceding the distractor
        or the complete sentence with the embedded distractor
        For explanations of ind and sent, see batch_eval

        Returns a string, dist's POS tag
        '''
        sent_with_dist = self.create_sent(ind,dist,sent,just_preceding)
        #check for comma earlier in the sentence - it gets its own tag
        for i in range(ind):
            if sent_with_dist.split()[i][-1] in string.punctuation:
                ind+=1
        return self.nlp(sent_with_dist)[ind].tag_
    #TODO: make this less specific to German, maybe a list of POS tags that should be capitalized?
    def pos_shennanigans(self,dists,ind,sent,tagged=True,compare_tag=None,just_preceding=False):
        '''
        Tags the distractors and uses these tags to determine whether a distractor should be capitalized
        If desired, also returns the tag with the appropriately capitalized distractor 

        Parameters
        dists = list of strings, the distractors we're going to evaluate
        For explanations of ind and sent, see batch_eval

        Optional
        tagged = boolean, whether or not keep track of the distractors' POS tags
        compare_tag = None or string, if None, then append any distractor, no matter its tag
        if a string that's an actual STTS tag, then distractors with this tag won't be appended, or ultimately evaluated
        compare_tag is the tag of the word we're trying to replace with a distractor, 
        so if indicated, distractors that are the same POS won't be considered
        See get_tag for an explanation of just_preceding

        Returns a list of strings (the distractors to be evaluated) or of tuples (str distractor, str tag) depending on tagged
        '''
        dists_final = []
        for dist in dists:
            tag = self.get_tag(dist,ind,sent,just_preceding)
            #This structure is not particularly generalizable - think of something smarter
            if (self.lang == "de" and (tag == 'NN' or tag == 'NE')) or (self.lang == "en" and (tag == "NNP" or tag == "NNPS")):
                new_dist = ""
                letter0 = dist[0].upper()
                new_dist = letter0 + dist[1:]
            else:
                new_dist = dist[:]
            if new_dist not in self.nonwords_set: #to prevent adding the capitalized version of a distractor that has actually already been evaluated
                if tagged and compare_tag != tag:
                    dists_final.append((new_dist,tag))
                elif not tagged:
                    dists_final.append(new_dist)
        return dists_final

    def adjust_params(self,params,temp_freqs,targets,finished_bin):
        '''Either move outward to still consider words in frequency bounds 
        or if we've evaluated all the distractors fitting the parameter settings,
        change the parameter settings in order to be able to evaluate enough distractors

        Parameters:
        params  = list of 2 floats: [upper bound of Zipf frequency to check, cdf_range]
        See get_len_range for explanation of cdf_range
        temp_freqs = list of two floats: [lower Zipf frequency key we're currently pulling distractors from, upper Zipf frequency key]
        They start at the same value, target_freq, and then the lower frequency is incrementally decreased 
        and the upper frequency incrementally increased until we hit the upper bound in params
        targets = tuple: (target_len,target_freq) 
        target_len = the length of the word we're finding a distractor for
        target_freq = the Zipf frequency of the same word

        Returns the updated (or not) params tuple and temp_freqs list
        '''
        new_params = params[:]
        new_tfs = temp_freqs[:]
        if finished_bin:
            new_tfs[0]-=0.01
            new_tfs[1]+=0.01
            #increase length range if we reach a certain freq distance
            if new_tfs[1] >= params[0]:
                new_params[0]+=0.1 
                if new_params[1] < 0.97: #can't increase above 1 
                    new_params[1]+=0.02 
                target_len,target_freq = targets
                #len_range = self.get_len_range(target_len,new_params[1])
                #to get longer or shorter words we didn't score last time
                new_tfs = [target_freq, target_freq]
        return new_params,new_tfs
    
    def get_rating(self,dist,ind,sent):
        '''
        Presents the distractor and the actual word similarly to how they'd be presented to a participant,
        then the experimenter can decide how easy it is to tell the distractor and actual word apart
        Entering a 1 means that the distractor is unacceptable in the context of the original sentence
        and therefore easily distinguishable from the actual word, making it a good distractor
        Entering a 0 means that a participant might think that the distractor is actually the next word in the sentence
        
        Parameters
        dist = string, the distractor we want the experimenter to rate
        For explanations of ind and sent, see batch_eval

        Returns a 0 or 1 rating entered by the experimenter, hopefully prescribing to the rating system described above
        '''
        rating = None
        print(f"{self.create_sent(ind,dist,sent,True)} [{sent[ind]}]") #Present the actual word alongside it
        try:
            rating = int(input("Unacceptable? ")) #will be either a 0 or 1
        except: #try try again
            print("Need to insert either 0 or 1")
            rating = int(input("Unacceptable? "))
        return rating

    def check_if_already_rated(self,dist_df,dist,ind,sent_ind,filler):
        print(dist_df)
        rating = None
        #look for rows with same ind, sent_ind, and filler columns
        already_rated = dist_df.loc[(dist_df["Sent_Index"] == sent_ind) & (dist_df["Index"] == ind) & (dist_df["Filler"] == filler)]
        if already_rated is not None:
            if dist in already_rated.Distractor.values:
                rating = already_rated[already_rated["Distractor"] == dist]["Rating"].values[0]
                if rating is None: #only possible non np types that rating can be are None (which throws an error) or ints (which don't)
                    rating = None
                elif np.isnan(rating):
                    rating = None
                else:
                    print(f"Distractor {dist} already rated {rating}")
        return rating
    
    def save_top_distractors(self,best,ind, sent, sent_ind, num_eval, filler, eval_time, rate=True, dist_df=None,dist_csv=None,save_csv=False,
                         just_preceding=True, dists_dict_file="PotentialDistractors.pkl",matching_dist=False):
        '''Save the best distractors that we found during evaluation in a dataframe and maybe also a csv

        Parameters:
        best = list of tuples, if tagged: (float PLL score, string distractor, string tag)
        else: float PLL score, string distractor)

        For explanations of ind, sent, sent_ind, num_eval, filler, and tagged see batch_eval

        Optional:
        rate = boolean, whether to ask the experimenter for a rating of whether the distractor works
        dist_df = pandas Dataframe, for the new data to be appended to
        dist_csv = string of csv path, converted to a DataFrame to which the new data is then appended
        save_csv = boolean, whether to save the resulting DataFrame in a csv in Drive, 
        can also pass in a string with a filename/path for saving the csv
        multi = boolean, whether we'll call this function multiple times in a row for the same sentence and distractor index
        in this case, we'll check if any of the same distractors that the experimenter already rated stayed in the top n,
        in which case we won't need to ask them to rate the distractor again

        Returns a DataFrame with the new distractor data
        '''
        already_rated = None
        if not dist_csv:
            if not isinstance(dist_df,pd.DataFrame):
            #if not dist_df: #original
                dist_df = pd.DataFrame(columns= ["Distractor","Index", "POS", "Sent_Index", "Filler", "Replaced_Word", "Replaced_POS", "PLL",
                                            "PLL_norm", "Number_Evaluated", "Best_N", "Ranking", "Rating","Evaluation_Time"])
        else:
            dist_df = pd.read_csv(f"{self.WORK_PATH}/{dist_csv}",index_col=0)
        best_n = len(best)
        ranking = best_n
        if isinstance(matching_dist,list):
            word1,word2 = matching_dist #assuming 2!
            word = f"{word1}/{word2}"
            replaced_tag = "N/A"
        elif matching_dist:
            word = sent[0][ind]
            replaced_tag = None
            #could just provide the first sentence though not using POS for anything
        else:
            word = sent[ind]
            #get rid of this is next iteration
            replaced_tag = self.get_tag(word,ind,sent,True)
        for i in range(best_n):
            if len(best[i]) == 3:
                pll,dist,tag = best[i]
            elif len(best[i]) == 2:
                pll,dist = best[i]
                tag = None
            if rate:
                rating = self.check_if_already_rated(dist_df,dist,ind,sent_ind,filler)
                if rating is None:
                    rating = self.get_rating(dist,ind,sent)
            else:
                rating = None
            new_row = [dist,ind,tag,sent_ind,filler,word,replaced_tag,pll,pll/len(dist),num_eval,best_n,ranking,rating,eval_time]
            dist_df.loc[len(dist_df)] = new_row
            ranking-=1
        if save_csv:
            if type(save_csv) == type(True):
                dist_df.to_csv(f"{self.WORK_PATH}/DistractorRecord.csv")
            elif isinstance(save_csv, str):
                dist_df.to_csv(f"{self.WORK_PATH}/{save_csv}")
            else:
                print(f"Invalid parameter type entered for save_csv. Saved to default path:{self.WORK_PATH}/DistractorRecord.csv")
                dist_df.to_csv(f"{self.WORK_PATH}/DistractorRecord.csv")
        #Save the distractors dictionary to which we've added new distractors 
        file_to_write = open(self.pickle_dict['dists_dict'], "wb")
        pickle.dump(self.dists_dict, file_to_write)
        return dist_df
    def batch_eval(self,ind,sent,sent_ind,num_eval,best_n,
               tagged=True,use_tag=False,filler=False,just_preceding=True,rate=True,
               dist_df=None,dist_csv=None,save_csv=False,verbose=False,matching_dist = False):
        ''' Evaluates specified number of appropriately matched distractor words 
            and saves the least probable of those distractors in the context of the orginal sentence 
            because they are the most likely to be the most obvious distractors to a participant

            Parameters:
            ind = int, index of the word we're replacing with a distractor
            sent = list of strings, words of the original sentence, in order
            sent_ind = int, index of the sentence in the list of experimental/filler sentences
            num_eval = int, number of distractors to evaluate from which to pull the top n
            best_n = int, the top n distractors to collect

            Optional:
            use_tag = boolean, whether to exclude distractors that have the same tag as the original word we're replacing
            filler = boolean, whether the sentence is an experimental item or a filler
            just_preceding = boolean, whether to get the sentence PLL score based on the sentence up to the distractor,
            otherwise insert the distractor into the complete sentence and then get the PLL score
            nonwords = set, words that are in the frequency dictionary but aren't actual German words, these should not be evaluated

            See save_top_distractors for explanations of rate, dist_df, dist_csv, save_csv
            See pos_shennanigans for an explanation of tagged

            Returns a pandas DataFrame with data of the best distractors
        '''
        start_time = time.time()
        #initializtion 
        evaluated = set()
        #num_dist = 0
        if isinstance(matching_dist,list):
            word1,word2 = matching_dist #assuming 2!
            if word1[-1] in string.punctuation: #should then be the same for word2
                word1 = word1[:-1]
            if word2[-1] in string.punctuation:
                word2 = word2[:-1]
            word = f"{word1}/{word2}"
            evaluated|=set(matching_dist)
        else:
            if matching_dist:
                word = sent[0][ind] #lower isn't necessary
            else:
                word = sent[ind]
            if word[-1] in string.punctuation: 
                word = word[:-1]
            evaluated.add(word)
        if use_tag:
            tag = self.get_tag(word,ind,sent,just_preceding)
            print(f"Word: {word}, Tag: {tag}")
            compare_tag = tag
        else:
            compare_tag = None
        #dont_eval = self.nonwords_set.copy() #to prevent mutation - necessary?
        if isinstance(matching_dist,list):
            target_len = (self.word_info[word1]["len"]+self.word_info[word2]["len"])/2
            target_freq = (self.word_info[word1]["freq"]+self.word_info[word2]["freq"])/2
        else:
            target_len = self.word_info[word]["len"]
            target_freq = self.word_info[word]["freq"]
        #print(target_freq)
        #print(target_len)
        targets = (target_len,target_freq)
        len_range = self.get_len_range(target_len)
        temp_freqs = [target_freq, target_freq]
        params = [target_freq+0.1, 0.2] #arbitrary - they're changed if we hit these thresholds so no reason to make them user-definable
        best_so_far = self.init_best_list(best_n)
        last_ind = 0
        finished_bin = True
        new_dists,num_dist = self.check_for_potential_distractors(word,num_eval) #has to do with this???
        if new_dists:
            if isinstance(matching_dist,list) or not matching_dist:
                #best_so_far,just_evaluated = eval_best_long(ind,new_dists,sent,scorer,best_so_far,tagged,just_preceding)
                best_so_far,just_evaluated = self.eval_best(ind,new_dists,sent,best_so_far,just_preceding)
            else:
                best_so_far,just_evaluated = self.eval_over_two_sents(ind,new_dists,sent,best_so_far,just_preceding)
            evaluated|=just_evaluated
        while num_dist < num_eval:
            #collect potential distractors based on frequency
            dist_temp = []
            freq_key0 = round(temp_freqs[0], 2) #to combat floating point problems
            if freq_key0 in self.freq_dict:
                bin_len = len(self.freq_dict[freq_key0])
                dist_temp+=self.freq_dict[freq_key0]
                #dist_temp = freq_dict[freq_key0] results in mutating freq_dict
            if num_dist:
                freq_key1 = round(temp_freqs[1], 2)
                if freq_key1 in self.freq_dict:
                    bin_len = len(self.freq_dict[freq_key1])
                    dist_temp+=self.freq_dict[freq_key1]
            #only keep potential distractors in length range
            #print(f"Length of dist_temp that we need to iterate through: {len(dist_temp)}")
            #dist_temp = [dist for dist in dist_temp if len_range[0]<=len(dist)<=len_range[1] and not dist[0].isdigit()]
            #already check for numbers in lang_spec code
            dist_temp = [dist for dist in dist_temp if len_range[0]<=len(dist)<=len_range[1]]
            #use a set to check that we haven't already evaluated these distractors
            #should prevent any other weird distractors we haven't already come across that are related to "00,00" and "00h"
            #dont_eval|=evaluated
            #new_dists = self.pos_shennanigans(set(dist_temp).difference(dont_eval),ind,sent,tagged,compare_tag,just_preceding) #could just check in here
            #check that the capitalized version of the distractor hasn't already been evaluated - not relevant anymore
            new_dists = list(set(dist_temp).difference(evaluated))
            if len(new_dists): #make sure it's not passing an empty list
                self.add_to_distractor_dict(word,new_dists,self.dists_dict)
                if len(new_dists)+num_dist > num_eval:#so as not to evaluate too many in case there are a lot of well-matched distractors
                    if isinstance(matching_dist,list) or not matching_dist:
                        #best_so_far,just_evaluated = eval_best_long(ind,new_dists,sent,scorer,best_so_far,tagged,just_preceding)
                        best_so_far,just_evaluated = self.eval_best(ind,new_dists[:num_eval-num_dist],sent,best_so_far,just_preceding)
                    else:
                        best_so_far,just_evaluated = self.eval_over_two_sents(ind,new_dists[:num_eval-num_dist],sent,best_so_far,just_preceding)
                    break
                if isinstance(matching_dist,list) or not matching_dist:
                    #best_so_far,just_evaluated = eval_best_long(ind,new_dists,sent,scorer,best_so_far,tagged,just_preceding)
                    best_so_far,just_evaluated = self.eval_best(ind,new_dists,sent,best_so_far,just_preceding)
                else:
                    best_so_far,just_evaluated = self.eval_over_two_sents(ind,new_dists,sent,best_so_far,just_preceding)
                evaluated|=just_evaluated #evaluated will always have 1 more than the actual number evaluated because the original word is added
                num_dist+=len(new_dists)
                if verbose:
                    print(f"new_dists: {new_dists}")
                    print(f"dist_temp: {dist_temp}")
                    #print(f"dont_eval: {dont_eval}")
                    print(f"best_so_far: {best_so_far}")
                    print(f"evaluated: {evaluated}")
            #may need to consider longer and more/less frequent words in order to evaluate enough distractors
            params,temp_freqs = self.adjust_params(params,temp_freqs,targets,finished_bin)
            len_range = self.get_len_range(target_len,params[1])

        eval_time = time.time()-start_time
        if verbose:
            print(f"Time to evaluate >{num_eval} distractors: {eval_time}")
            print(f"Final parameter settings: {params[0]} {params[1]}")
        return self.save_top_distractors(best_so_far,ind,sent,sent_ind,num_eval,filler,eval_time,rate,dist_df,dist_csv,save_csv,just_preceding,matching_dist=matching_dist)

    def check_for_potential_distractors(self,word_to_replace,num_eval):
        if word_to_replace in self.dists_dict:
            new_dists = list(self.dists_dict[word_to_replace]) #stored in the dictionary as a set
            new_dists_len = len(new_dists)
            if new_dists_len > num_eval:
                new_dists = new_dists[:num_eval+1]
                num_dist = num_eval
            elif new_dists_len == num_eval:
                num_dist = num_eval
            else:
                num_dist = new_dists_len
            return new_dists,num_dist
        else:
            return [],0

    def add_to_distractor_dict(self,word_to_replace,new_dists,dist_dict):
        dists_set = set(new_dists)
        if word_to_replace not in dist_dict:
            dist_dict[word_to_replace] = dists_set
        else:
            dist_dict[word_to_replace]|=dists_set
        return dist_dict
    
    def add_ratings(self,dist_df,exp_sents_list,filler_sents_list,max_rate=None):
        '''
        Prompt the experimenter to rate the distractors in dist_df that haven't been rated yet

        Parameters
        dist_df = pandas DataFrame of distractors, some of which are missing experimenter ratings
        exp_sents_list = list of lists of strings, corresponding to the experimental items
        filler_sents_list = list of lists of strings, corresponding to the filler sentences
        Neither of these lists should ever have their ordering changed

        Returns the updated DataFrame, whose distractors are now all rated
        '''
        '''num_rate = len(dist_df)
        if max_rate:
            num_rate = min(len(dist_df),max_rate) #too simple'''
        if not max_rate:
            max_rate = len(dist_df)
        num_rated = 0
        for i in range(len(dist_df)):
            row = dist_df.iloc[i]
            if row["Rating"] is None or np.isnan(row["Rating"]):
                ind = row["Index"]
                dist = row["Distractor"]
                sent_ind = row["Sent Index"]
                filler = row["Filler"]
                rating = self.check_if_already_rated(dist_df,dist,ind,sent_ind,filler)
                if rating is None:
                    if row["Filler"]:
                        sent = filler_sents_list[row["Sent Index"]]
                    else:
                        sent = exp_sents_list[row["Sent Index"]]
                        rating = self.get_rating(dist,ind,sent)
                        num_rated+=1
                dist_df.at[i,"Rating"] = rating
                if num_rated >= max_rate:
                    break
        return dist_df

    def compare_batches(self,batches,sent,sent_ind,best_n,start_ind=1,end_ind=None,tagged=True,use_tag=False,
                    filler=False,just_preceding=True,rate=True,dist_df=None,dist_csv=None,save_csv=True,matching_dist=False):
        '''
        Collects data of distractors from multiple rounds of evaluating different numbers of distractors
        for all the words in the passed sentence

        Parameters
        batches = list of ints, the number of distractors that should be evaluated for each batch
        
        Optional
        start_ind = int, the index of the first word in the sentence that we want replaced with a distractor, 
        mostly just changed in testing contexts

        end_ind = None or int, if None, we'll find distractors for words from start_ind to the end of the sentence,
        otherwise, end_ind will be the index of the last word in the sentence that we will find distractors for, 
        also mostly just changed in testing contexts

        See batch_eval for explanations of any of the other parameters

        Returns a pandas DataFrame with distractor data for the entire sentence (or for words from start_ind to end_ind)
        '''
        if matching_dist:
            end_ind = len(sent[0])
        elif not end_ind:
            end_ind = len(sent)
        elif start_ind < 0 or end_ind < start_ind or end_ind > len(sent):
            raise ValueError("Please correct start_ind and/or end_ind.")
        first = True
        num_rounds = (end_ind-start_ind)*len(batches)
        round = 1
        #snapshot0 = tracemalloc.take_snapshot()
        for ind in range(start_ind,end_ind):
            for num_eval in batches:
                #Could be part of verbose
                #print(f"Round {round}/{num_rounds}: Find distractor for word at index {ind} by comparing {num_eval} potential distractors")
                #snapshot1 = tracemalloc.take_snapshot()
                save = False
                if num_eval > 100 or round == num_rounds:
                    if isinstance(save_csv, str):
                        save = save_csv
                    else:
                        save = True
                if first:
                    temp_df = self.batch_eval(ind,sent,sent_ind,num_eval,best_n,tagged,use_tag,filler,just_preceding,rate,dist_df,dist_csv,save,matching_dist=matching_dist)
                    first = False
                else:
                    temp_df = self.batch_eval(ind,sent,sent_ind,num_eval,best_n,tagged,use_tag,filler,just_preceding,rate,temp_df,save_csv=save,matching_dist=matching_dist) 
                round+=1
        return temp_df

            