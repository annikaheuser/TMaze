import pandas as pd
import re
import string

def sents_differ(item_dict):
  sents = []
  for pair in item_dict.items():
    sents.append(pair[1].split())
  sent1,sent2 = sents #assuming just 2, can make this more generalizable later
  for i in range(len(sent1)):
    if sent1[i] != sent2[i]:
      start = sent1[:i]
      diff_words = [sent1[i],sent2[i]]
      full_sents = [sent1,sent2]
      return start,diff_words,full_sents 

def match_sent(tmaze_obj,sent,sent_ind,num_eval,best_n,filler,tagged=True,save_df=True):
    '''first_dist = "X"
    while len(first_dist) < len(sent[0]):
        first_dist+="x"'''
    js_distractors = "a:\"x-x-x "
    if type(save_df) == type(True):
        dist_df = tmaze_obj.compare_batches([num_eval],sent,sent_ind,best_n,filler=filler,tagged=tagged,rate=False)
    else: #can look for a pandas df more specifically for error handling later
        dist_df = tmaze_obj.compare_batches([num_eval],sent,sent_ind,best_n,filler=filler,tagged=tagged,rate=False,dist_df=save_df)
    dists = dist_df.loc[(dist_df["Sent_Index"] == sent_ind) & (dist_df["Filler"] == filler) & (dist_df["Ranking"] == 1) & (dist_df["Number_Evaluated"] == num_eval)].Distractor
    for dist,i in zip(dists,range(1,len(sent))):
        js_distractors+=dist
        if sent[i][-1] in string.punctuation: #may want to consider inserting the distractor with the comma
            js_distractors+=sent[i][-1]
        js_distractors+=" "
    if type(save_df) == type(True):
        if save_df:
            return js_distractors[:-1]+"\"",dist_df
        else:
            return js_distractors[:-1]+"\""
    return js_distractors[:-1]+"\"",dist_df

def make_item(sent_ind,cond,sent,dist_sent):
  sent_id = f"[\"{cond}\", {int(sent_ind)}],"
  content = f"{{s:\"{sent}\", {dist_sent}}}"
  sent_item = f"[{sent_id} \"Maze\", {content}],"
  return sent_item

def create_item_groups(tmaze_obj,sent_ind,item_dict,num_eval,best_n,save_df=True):
  filler = False
  match_many = True
  conds = list(item_dict.keys())
  if len(conds) == 1:
    filler = True
    match_many = False
  sent_items= ""
  if match_many:
    sent_list1,diff_words,full_sents = sents_differ(item_dict)
    dist_sent, dist_df = match_sent(tmaze_obj,sent_list1,sent_ind,num_eval,best_n,filler,save_df=save_df)
    dist_sent = dist_sent[:-1] #get rid of ending double quote
    ind = len(sent_list1)
    dist_df = tmaze_obj.batch_eval(ind,full_sents[0],sent_ind,100,2,matching_dist=diff_words,rate=False,dist_df=dist_df)
    best_dist = dist_df.loc[(dist_df["Ranking"] == 1) & (dist_df["Index"] == ind) & (dist_df["Sent_Index"] == sent_ind)].Distractor.values[0]
    punct_best_dist = None
    if diff_words[0][-1] in string.punctuation:
      if diff_words[1][-1] not in string.punctuation:
        punct_best_dist = best_dist
        dist_sent1 = dist_sent+" "+punct_best_dist+" "
      best_dist+=diff_words[0][-1]
    elif diff_words[1][-1] in string.punctuation: #won't trigger if the first one has punctuation
      punct_best_dist = best_dist+diff_words[1][-1]
      dist_sent1 = dist_sent+" "+punct_best_dist+" "
    dist_sent = dist_sent+" "+best_dist+" "
    dist_df = tmaze_obj.compare_batches([100],full_sents,sent_ind,best_n,ind+1,filler=filler,rate=False,dist_df=dist_df,matching_dist=True)
    chosen_dists = dist_df.loc[(dist_df["Ranking"] == 1) & (dist_df["Sent_Index"] == sent_ind)]
    for i in range(ind+1,len(full_sents[0])):
        to_add = ""
        next_dist = chosen_dists.loc[chosen_dists["Index"] == i].Distractor.values[0]
        to_add+=next_dist
        if i < len(full_sents[0])-1:
            to_add+=" "
        else:
            to_add+=full_sents[0][-1][-1] #ending punctuation
            to_add+='"'
        dist_sent+=to_add
        if punct_best_dist:
            dist_sent1+=to_add
    for i in range(len(conds)):
      if i and punct_best_dist:
        alt_sent = dist_sent1
      else:
        alt_sent = dist_sent
      #except for comma case dist_sent is the same for both - that's kinda the point
      sent = item_dict[conds[i]]
      sent_items+=make_item(sent_ind,conds[i],sent,alt_sent)
      sent_items+="\n"
    sent_items = sent_items[:-1]
  else:
    cond = conds[0]
    sent = item_dict[cond]
    dist_sent, dist_df = match_sent(tmaze_obj,sent.split(),sent_ind,num_eval,best_n,filler,save_df=save_df)
    sent_items+=make_item(sent_ind,cond,sent,dist_sent)
  return sent_items,dist_df

def replace_disractors_in_item(item,weird_dists,dist_df,matched_dists):
    re_dist_sent = '(?:a:")(.+)(?:"})'
    re_item_num = "[0-9]+"
    re_sent_start = ".+(?=(a:))"
    item_start = re.search(re_sent_start,item)[0]
    dist_sent = re.search(re_dist_sent,item)[1]
    sent_ind = re.search(re_item_num,item)[0]
    if type(matched_dists) == dict:
        if sent_ind in matched_dists:
            return matched_dists[sent_ind]
    dist_sent_list = dist_sent.split()
    new_sent = item_start+'a: "'
    for og_word,ind in zip(dist_sent_list,range(len(dist_sent_list))):
        punct = False
        if og_word[-1] in string.punctuation:
            word = og_word[:-1]
            punct = True
        else:
            word = og_word
        new_word = og_word
        if word in weird_dists:
            other_dists = dist_df.loc[(dist_df["Sent_Index"] == sent_ind) & (dist_df["Index"] == ind)]
            for rank in range(2,len(other_dists)+1):
                next_dist = other_dists.loc[other_dists["Ranking"] == rank].Distractor.values[0]
                if next_dist not in weird_dists: #whoo don't need to search anymore
                    new_word = next_dist
                    if punct:
                        new_word+=og_word[-1]
                    break
        new_sent+=new_word
        new_sent+=" "
    new_sent=new_sent[:-1]+'"}],\n'
    if type(matched_dists) != type(False):
        matched_dists[sent_ind] = new_sent 
    return new_sent

def replace_weird_distractors(js_file,new_js_file,weird_dists,dist_df,matched_dists=False):
    #may want to get rid of them from the freq_bins and add them to the nonwords set too
    js_to_write = ""
    if isinstance(dist_df, str): #otherwise assuming we already have a DataFrame
        dist_df = pd.read_csv(dist_df)
    with open(js_file) as f:
        exp_items = f.readlines()
    matched_dict = False
    if matched_dists:
        matched_dict = {}
    for item in exp_items:
        new_sent = replace_disractors_in_item(item,weird_dists,dist_df,matched_dict)
        js_to_write+=new_sent
    with open(new_js_file,"w") as f:
        f.write(js_to_write)

def compile_all_sent_items_from_dict(tmaze_obj,item_dicts,num_eval,best_n):
  all_sents_js = ""
  dist_df = True
  ind = 0
  for key in item_dicts:
    sent_item,dist_df = create_item_groups(tmaze_obj,key,item_dicts[key],num_eval,best_n,dist_df)
    #print(sent_item)
    all_sents_js+=sent_item
    all_sents_js+="\n"
    '''if ind > 5:
      break'''
    ind+=1
    print(f"Finished item {ind} of {len(item_dicts)}")
  return all_sents_js,dist_df
        
        
            
                



    

