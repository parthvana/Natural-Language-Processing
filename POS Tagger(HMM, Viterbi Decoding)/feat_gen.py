#!/bin/python
#from gensim.models import Word2Vec
import pickle
#from gensim.models import Word2Vec
#from itertools import chain
#from collections import OrderedDict
#from collections import Counter
#import nltk
##from nltk.corpus import brown
##from nltk import ConditionalFreqDist
##nltk.download('treebank') 
##nltk.download("universal")
#from sklearn.cluster import KMeans




def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    
    
#    nltk.download("universal")
#    nltk.download('treebank')
  
  
#    #Clustering on Gensim Word2vec
#         
#    google_news=Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
#    google_news.save("gn_vectors")
#    train = pickle.load(open("../data/twitter_train.p", "rb"))
#    new_docs = list(chain(*train))
#    vocab = list(set(new_docs))
#    gensim_corpus = Word2Vec.load("gensim_100000")
#    corpus_vectors = OrderedDict()
#    index_corpus = OrderedDict()
#    
#    i = 0
#    
#    for d in vocab:
#     	 if d not in corpus_vectors:
#              try:
#                  corpus_vectors[d] = gensim_corpus[d.strip().lower()]
#                  index_corpus[i] = d
#                  i += 1
#              except Exception, e:
#                  pass
# 
#    pickle.dump(corpus_vectors, open("word_train_data.p", "wb"))
#    pickle.dump(index_corpus, open("index_train_data.p", "wb"))
#    
#
#     #Clustering
#    
#    corpus_vectors = pickle.load(open("word_train_data.p", "rb"))
#    index_corpus = pickle.load(open("index_train_data.p", "rb"))
#    corpus_matrix = corpus_vectors.values()
#    lbls = kmeans.labels_
#    corpus_clst = OrderedDict() 
#    n_clust = 30
#    kmeans_clustering = KMeans(n_clusters=n_clust)
#    kmeans = kmeans_clustering.fit(corpus_matrix)
#    
#    
#    
#    for i in range(len(lbls)):
#        corpus_clst[index_corpus[i]] = lbls[i]
#    
#    list_corpus = {}
#    
#    for j in range(len(lbls)):
#        try:
#            list_corpus[lbls[j]].append(index_corpus[j])
#        except Exception:
#            list_corpus[lbls[j]] = [index_corpus[j]]
##   for j in xrange(L):
##            start_scores[j] = w[0,self.get_start_trans_idx(j)]
##            end_scores[j] = w[0,self.get_end_trans_idx(j)]
##            # transition
##            for k in xrange(L):
##                trans_scores[j][k] = w[0,self.get_trans_idx(j, k)]
##            # emission
##            for i in xrange(N):
##                score = 0.0
##                for fidx in X[i]:
##                    score += w[0,self.get_ftr_idx(fidx, j)]
##                emission_scores[i][j] = score
##    
#        
#    pickle.dump(word_cluster, open("final_word_cluster.p", "wb"))
    
    
    pass

fdr = open("cluster_word2vec.p", 'rb')  
word2vec_cluster = pickle.load(fdr)  
fdr.close()

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
#   ftrs = self.token2features(sent, i)
#        fidxs  = []
#        for ftr in ftrs:
#            if ftr in self.fmap:
#                fidxs.append(self.get_index(ftr))
    
    
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    dict=word
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    
#    assert self.frozen == False
#        if ftr not in self.fmap:
#            fidx = len(self.feats)
#            self.fmap[ftr] = fidx
#            self.feats.append(ftr)
#            if self.num_features % 1000 == 0:
#                print "--", self.num_features, "features added."
#            self.num_features = len(self.feats)
    
    #General Heuristics
    if(dict[0].isupper()==True):
        ftrs.append("P_N")
    
    # Features for suffixes
    
    if dict.lower().strip()[-3:]=="acy" or dict.lower().strip()[-4:]=="ance" or dict.lower().strip()[-4:]=="ence" or  dict.lower().strip()[-3:]=="ist" or dict.lower().strip()[-4:]=="sion" or dict.lower().strip()[-4:]=="tion" or dict.lower().strip()[-4:]=="ship" or dict.lower().strip()[-3:]=="ist" or dict.lower().strip()[-3:]=="ism" or dict.lower().strip()[-3:]=="dom" or dict.lower().strip()[-4:]=="ment" or dict.lower().strip()[-3:]=="ity" :
         ftrs.append("P_N")
    
    if(len(dict)>4 and dict.lower().strip()[-2:]=="er"):
        ftrs.append("P_N")
    
    if(dict[-2:]=="ly"):
        ftrs.append("P_ADV")
        
    if (len(dict) > 4 and dict.lower().strip()[-2:] == "or"):
        ftrs.append("P_N")
    
    if dict.lower().strip()[-3:]=="ate" or dict.lower().strip()[-2:]=="en" or dict.lower().strip()[-2:]=="ed" or dict.lower().strip()[-3:]=="ize" or dict.lower().strip()[-3: ]=="ing":  
        ftrs.append("P_VB")
        ftrs.append("P_N")
        
        
    if(dict[-2:]=="ly") or (dict[-4:]=="ward") or (dict[-2:]=="ly") or (dict[-2:]=="wise"):
        ftrs.append("P_ADV")
#    corpus_vectors = pickle.load(open("word_train_data.p", "rb"))
#    index_corpus = pickle.load(open("index_train_data.p", "rb"))
#    corpus_matrix = corpus_vectors.values()
#    lbls = kmeans.labels_
#    corpus_clst = OrderedDict() 
    if len(dict)>5 and dict.lower().strip()[-3:]=="ish":
        ftrs.append("P_ADJ")
   
    if dict[-4:].lower().strip() == "able" or dict.lower().strip()[-4:] == "ious" or dict.lower().strip()[-4:] == "ible" or dict.lower().strip()[-4:] == "ical" or dict.lower().strip()[-3:]=="ish" or dict.lower().strip()[-4:]=="less" or dict.lower().strip()[-3:]=="ful" or dict.lower().strip()[-2:]=="ic" or dict.lower().strip()[-3:]=="ive" or dict.lower().strip()[-5:]=="esque" or dict.lower().strip()[-2:]=="al" :
        ftrs.append("P_ADJ")

    if(len(dict.lower().strip())>5):
        if dict.lower().strip()[-3:] == "ize"or dict.lower().strip()[-3:] == "ise":
            ftrs.append("P_VB")
    
    #Features indicating punctuation mark or end of statement
    p_m=[".","?","?!","!","."]
    for l in p_m:
        
        if dict[0]==l and dict[1:].isalpha==False:
            ftrs.append(("P_T"))
    
    #Features specific to Social Twitter Data - emoticons and punctuation mark and website
    emo_sym = [":-)",  "=)",
                ":-D", ":)",":D", "8-D","x-D", "xD", "X-D", "XD",
                ":-(", ":(",
                ":-|", "8D",":|",
                ":'-(", ":'()",
                ":'-)", ":')",
                ":-o", ":-O", ":o", ":O",
                "o_O", "o.O", "O_o", "O.o",
                ":*", ";-)", ";)",
                "%-)",
                "<3", "</3"];
    for k in emo_sym:
        if dict==k:
            ftrs.append(("P_T"))

    if("www." in dict.lower().strip() or ".com" in dict.lower().strip() or "http" in dict.lower().strip() or ".co" in dict.lower().strip()):
        ftrs.append(("P_T"))
    
    if(dict[0]=="@" or dict[0]=="#"):
        ftrs.append(("P_T"))
    
#    #Brown Clustering
#    corpus = nltk.corpus.treebank.tagged_words(tagset='universal')
#    freq_dist = nltk.ConditionalFreqDist(corpus)
#    try:
#         if(freq_dist[dist.lower().strip()].most_common()!=[]):
#             ftrs.append("Brown"+str(freq_dist[word.lower().strip()].most_common()[0][0].strip()))
#    except Exception:
#         pass

     
    if(dict.lower().strip() in word2vec_cluster):
        ftrs.append("Cls"+str(word2vec_cluster[dict.lower().strip()]))


    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "sleep", "more", "than", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print(sent[i], ":", token2features(sent, i))
