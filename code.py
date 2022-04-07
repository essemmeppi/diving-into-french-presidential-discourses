#########################################################
#########################################################


## Diving into French Presidential Discourses

## Python script of our research project for the course 'Diving into Digital Public Spheres' (Sciences Po, CEU)
## Authors: Giulia Annaliese Paxton, Ryan Hachem, and Simone Maria Parazzoli
## Date: 07/04/2022


#########################################################
#########################################################


!pip install minet
!pip install pdfminer.six
!pip install unidecode
!python -m spacy download fr_core_news_sm

import pandas as pd
from pdfminer.high_level import extract_text
from gensim.test.utils import datapath
from gensim import utils
import spacy
from collections import Counter
import unidecode
import gensim.models


#########################################################
#########################################################


## Emmanuel Macron (EM)


## create a .csv of EM's Twitter activity since 27 Jan 2022
!minet tw scrape tweets "(from:EmmanuelMacron) until:2022-03-27 since:2022-01-27" > tweets_EM.csv

## convert the .csv file in a data frame using pandas
df_tw_EM = pd.read_csv("./tweets_EM.csv")

## create a list of tweets selecting the 'text' column of the data frame
list_tw_EM = df_tw_EM['text'].values.tolist()
len(list_tw_EM)

## EM affiliates' twitter activity
!minet tw scrape tweets "(from:RolandLescure OR from:ilanacicurelrem OR from:ebothorel OR from:mguevenoux OR from:StanGuerini OR from:JulienBargeton OR from:Ambroise_Mejean OR from:RichardFerrand OR from:MaudBregeon OR from:LauStmartin OR from:cedric_o OR from:JeanCASTEX OR from:franckriester OR from:BrunoLeMaire OR from:AgnesRunacher) until:2022-03-27 since:2022-01-27" > tw_EM_aff_all.csv

## convert .csv affiliates' tweets in a list
df_tw_EM_aff_all = pd.read_csv("tw_EM_aff_all.csv")
list_tw_EM_aff_all = df_tw_EM_aff_all['text'].values.tolist()
print(list_tw_EM_aff_all[0])
print(len(list_tw_EM_aff_all))

## merge EM's and EM's affiliates lists
list_tw_EM_all = list_tw_EM + list_tw_EM_aff_all

## i retrieve a string from the pdf of EM's manifesto using extract_text of the pdfminer package
## the cleaning process is specific for this manifesto and it depends on the output of extract_text
manif_EM = extract_text('/Users/simonemariaparazzoli/Documents/Università/Sciences Po/Diving into public digital spaces/research/manifesto_macron.pdf')
manif_clean_EM = manif_EM.replace('-\n','')
manif_clean_EM = manif_clean_EM.replace('\n','')
manif_clean_EM = manif_clean_EM.replace('\xa0',' ')
manif_clean_EM = manif_clean_EM.replace('\x0c',' ')
manif_clean_EM = manif_clean_EM.replace('.','---')
#print(repr(manif_clean_EM))

## convert the string of the manifesto into a list
list_manif_EM = manif_clean_EM.split("---")
list_manif_EM = [s for s in list_manif_EM if len(s)>20]
len(list_manif_EM)
#print(list_manif_EM)

## merge the two lists of tweets and of the manifesto 
list_EM = list_tw_EM_all + list_tw_EM_all + list_manif_EM
len(list_EM)

## load a spacy model to retrieve stop words
nlp = spacy.load("fr_core_news_sm")
stop_words_fr = nlp.Defaults.stop_words
#new_sw = ["avec","la","les","le","pour","un","une","nous","ete","et","je"]
#stop_words_fr.add(new_sw)

## clean the list of tweets and manifesto to get rid of useless words and make the list content homogeneous
list_EM_clean = []
for i in list_EM:
    doc = nlp(i)
    tokens = [unidecode.unidecode(token.text).lower()for token in doc 
              if ( token.text not in stop_words_fr and
                  len(token.text)>1 and
                  token.like_url == False )]
    tokens_joined = ' '.join(tokens)
    list_EM_clean.append(tokens_joined)
    
## test the output of the cleaning process
print(list_EM[2401])
print("---")
print(list_EM_clean[2401])

## prepare the corpus as a class
class MyCorpus_EM:

    def __iter__(self):
        for i in list_EM_clean:
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(i,min_len=3)

## train the word embeddings model_EM
sentences = MyCorpus_EM()
model_EM = gensim.models.Word2Vec(sentences=sentences, min_count=10, vector_size=300, epochs=100)

## transform the corpus list (that is made of tweets and sentences from the manifesto)
## in a list containing all the words of the corpus as elements of the list
words_EM = []

for i in list_EM_clean:
    i_split = i.split(' ') #transform the i document into a list (split at blank space)
    words_EM.extend(i_split)

## clean the list of tokens
words_EM_clean = [x for x in words_EM 
                   if x not in stop_words_fr
                   if x != "\n\n"
                   if len(x)>1]

## find the 30 most common words using Counter
words_freq_EM = Counter(words_EM_clean)
common_words_EM = words_freq_EM.most_common(100)
print(common_words_EM)

## finding the most similar words in the model
result = model_EM.wv.most_similar(positive=['france'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['etat'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['souverainete'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['president'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['politique'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['droit'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['entreprise'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['economie'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['emploi'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['travail'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['numerique'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['donnees'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['monde'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['realite'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['societe'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['pouvoir'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['avenir'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['histoire'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['contre'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['faut'], topn=20)
print(result)

result = model_EM.wv.most_similar(positive=['crise'], topn=20)
print(result)


#########################################################
#########################################################


## Eric Zemmour (EZ)


## create a .csv of EZ's Twitter activity since 27 Jan 2022
!minet tw scrape tweets "(from:ZemmourEric) until:2022-03-27 since:2022-01-27" > tweets_EZ.csv

## convert the .csv file in a data frame using pandas
df_tw_EZ = pd.read_csv("./tweets_EZ.csv")

## create a list of tweets selecting the 'text' column of the data frame
list_tw_EZ = df_tw_EZ['text'].values.tolist()
len(list_tw_EZ)

## retrieve EZ's affiliates Twitter activity
!minet tw scrape tweets "(from:Samuel_Lafont OR from:GilbertCollard OR from:jerome_riviere OR from:MarionMarechal OR from:G_Peltier OR from:NicolasBay_ OR from:DenisCieslik OR from:stanislasrig OR from:AntoineDiers OR from:de_beaujeu OR from:Stephane_Ravier OR from:MaxettePirbakas OR from:LaurenceTrochu) until:2022-03-27 since:2022-01-27" > tw_EZ_aff_all.csv

## convert EZ's affiliates' tweets in a list
df_tw_EZ_aff_all = pd.read_csv("tw_EZ_aff_all.csv")
list_tw_EZ_aff_all = df_tw_EZ_aff_all['text'].values.tolist()
print(list_tw_EZ_aff_all[0])
print(len(list_tw_EZ_aff_all))

## merge EZ and his affiliates' lists
list_tw_EZ_all = list_tw_EZ + list_tw_EZ_aff_all

## i retrieve a string from the pdf of EZ's manifesto using extract_text of the pdfminer package
## the cleaning process is specific for this manifesto and it depends on the output of extract_text
manif_EZ = extract_text('/Users/simonemariaparazzoli/Documents/Università/Sciences Po/Diving into public digital spaces/research/manifesto_zemmour.pdf')
manif_clean_EZ = manif_EZ.replace('-\n','')
manif_clean_EZ = manif_clean_EZ.replace('\n\n','---')
manif_clean_EZ = manif_clean_EZ.replace('\n','')
manif_clean_EZ = manif_clean_EZ.replace('\xa0','')
manif_clean_EZ = manif_clean_EZ.replace('\x0c','')
manif_clean_EZ = manif_clean_EZ.replace('. .','')
manif_clean_EZ = manif_clean_EZ.replace('  ','')
manif_clean_EZ = manif_clean_EZ.replace('. ','---')
manif_clean_EZ = manif_clean_EZ.replace('------','---')
#print(repr(manif_clean_EZ))

## convert the string of the manifesto into a list
list_manif_EZ = manif_clean_EZ.split("---")
list_manif_EZ = [s for s in list_manif_EZ if len(s)>30]
len(list_manif_EZ)
#print(list_manif_EZ)

## merge the two lists of tweets and of the manifesto 
list_EZ = list_tw_EZ_all + list_manif_EZ
len(list_EZ)

## load a spacy model to retrieve stop words
nlp = spacy.load("fr_core_news_sm")
stop_words_fr = nlp.Defaults.stop_words

## clean the list of tweets and manifesto to get rid of useless words and make the list content homogeneous
list_EZ_clean = []
for i in list_EZ:
    doc = nlp(i)
    tokens = [unidecode.unidecode(token.text).lower() for token in doc 
              if (token.text not in stop_words_fr and
                  len(token.text)>2 and
                  token.like_url == False )]
    tokens_joined = ' '.join(tokens)
    list_EZ_clean.append(tokens_joined)
    
## test the output of the cleaning process
print(list_EZ[205])
print("---")
print(len(list_EZ_clean))

## prepare the corpus as a class
class MyCorpus_EZ:

    def __iter__(self):
        for i in list_EZ_clean:
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(i,min_len=3)
            
## train the word embeddings model_EZ
sentences = MyCorpus_EZ()
model_EZ = gensim.models.Word2Vec(sentences=sentences, min_count=10, vector_size=300, epochs=100)

## transform the corpus list (that is made of tweets and sentences from the manifesto)
## in a list containing all the words of the corpus as elements of the list
words_EZ = []

for i in list_EZ_clean:
    i_split = i.split(' ') #transform the i document into a list (split at blank space)
    words_EZ.extend(i_split)

## clean the list of tokens
words_EZ_clean = [x for x in words_EZ 
                   if x not in stop_words_fr
                   if x != "\n\n"
                   if len(x)>1]

## find the 30 most common words using Counter
words_freq_EZ = Counter(words_EZ_clean)
common_words_EZ = words_freq_EZ.most_common(30)
print(common_words_EZ)

result = model_EZ.wv.most_similar(positive=['france'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['etat'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['souverainete'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['president'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['politique'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['droit'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['entreprise'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['economie'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['emploi'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['travail'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['taxes'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['numerique'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['monde'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['realite'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['verite'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['societe'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['pouvoir'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['avenir'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['histoire'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['contre'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['faut'], topn=20)
print(result)

result = model_EZ.wv.most_similar(positive=['crise'], topn=20)
print(result)


#########################################################
#########################################################


## Marine Le Pen (MLP)


## create a .csv of MLP's Twitter activity since 27 Jan 2022
!minet tw scrape tweets "(from:MLP_officiel) until:2022-03-27 since:2022-01-27" > tweets_MLP.csv

## convert the .csv file in a data frame using pandas
df_tw_MLP = pd.read_csv("./tweets_MLP.csv")

## create a list of tweets selecting the 'text' column of the data frame
list_tw_MLP = df_tw_MLP['text'].values.tolist()
len(list_tw_MLP)

## retrieve MLP's affiliates Twitter activity
!minet tw scrape tweets "(from:de_beaujeu OR from:JulienOdoul OR from:sebchenu OR from:SteeveBriois OR from:ljacobelli OR from:jsanchez_rn OR from:jllacapelle OR from:david_rachline OR from:franckallisio OR from:ThierryMARIANI OR from:BallardPhilippe OR from:louis_aliot OR from:wdesaintjust OR from:BrunoBilde) until:2022-03-27 since:2022-01-27" > tw_MLP_aff_all.csv

## convert MLP's affiliates' tweets in a list
df_tw_MLP_aff_all = pd.read_csv("tw_MLP_aff_all.csv")
list_tw_MLP_aff_all = df_tw_MLP_aff_all['text'].values.tolist()
print(list_tw_MLP_aff_all[0])
print(len(list_tw_MLP_aff_all))

## merge MLP and his affiliates' lists
list_tw_MLP_all = list_tw_MLP + list_tw_MLP_aff_all
print(len(list_tw_MLP_all))

## i retrieve a string from the pdf of MLP's manifesto using extract_text of the pdfminer package
## the cleaning process is specific for this manifesto and it depends on the output of extract_text
manif_MLP = extract_text('/Users/simonemariaparazzoli/Documents/Università/Sciences Po/Diving into public digital spaces/research/manifesto_lepen.pdf')
manif_clean_MLP = manif_MLP.replace('-\n','')
manif_clean_MLP = manif_clean_MLP.replace('\n\n',' ')
manif_clean_MLP = manif_clean_MLP.replace('\n','')
manif_clean_MLP = manif_clean_MLP.replace('\uf0e8\u2009','---')
manif_clean_MLP = manif_clean_MLP.replace('\uf0e8\xa0','---')
manif_clean_MLP = manif_clean_MLP.replace('\x0c',' ')
manif_clean_MLP = manif_clean_MLP.replace('\xa0','')
#print(repr(manif_clean_MLP))

## convert the string of the manifesto into a list
list_manif_MLP = manif_clean_MLP.split("---")
list_manif_MLP = [s for s in list_manif_MLP if len(s)>30]
len(list_manif_MLP)
#print(list_manif_MLP)

## merge the two lists of tweets and of the manifesto 
list_MLP = list_tw_MLP_all + list_tw_MLP_all + list_manif_MLP
len(list_MLP)

## load a spacy model to retrieve stop words
nlp = spacy.load("fr_core_news_sm")
stop_words_fr = nlp.Defaults.stop_words

## clean the list of tweets and manifesto to get rid of useless words and make the list content homogeneous
list_MLP_clean = []
for i in list_MLP:
    doc = nlp(i)
    tokens = [unidecode.unidecode(token.text).lower() for token in doc 
              if (token.text not in stop_words_fr and
                  len(token.text)>2 and
                  token.like_url == False )]
    tokens_joined = ' '.join(tokens)
    list_MLP_clean.append(tokens_joined)
    
## test the output of the cleaning process
print(list_MLP[205])
print("---")
print(list_MLP_clean[205])

## prepare the corpus as a class
class MyCorpus_MLP:

    def __iter__(self):
        for i in list_MLP_clean:
            yield utils.simple_preprocess(i,min_len=3)
            
## train the word embeddings model_MLP
sentences = MyCorpus_MLP()
model_MLP = gensim.models.Word2Vec(sentences=sentences, min_count=10, vector_size=300, epochs=100)

## transform the corpus list (that is made of tweets and sentences from the manifesto)
## in a list containing all the words of the corpus as elements of the list
words_MLP = []

for i in list_MLP_clean:
    i_split = i.split(' ') #transform the i document into a list (split at blank space)
    words_MLP.extend(i_split)

## clean the list of tokens
words_MLP_clean = [x for x in words_MLP 
                   if x not in stop_words_fr
                   if x != "\n\n"
                   if len(x)>1]

## find the 30 most common words using Counter
words_freq_MLP = Counter(words_MLP_clean)
common_words_MLP = words_freq_MLP.most_common(30)
print(common_words_MLP)

## first attempt with the most_similar function on our corpus using our model_MLP
result = model_MLP.wv.most_similar(positive=['france'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['etat'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['souverainete'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['president'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['politique'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['droit'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['entreprise'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['economie'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['emploi'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['travail'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['taxes'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['monde'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['realite'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['verite'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['societe'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['pouvoir'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['avenir'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['histoire'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['contre'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['faut'], topn=20)
print(result)

result = model_MLP.wv.most_similar(positive=['crises'], topn=20)
print(result)


#########################################################
#########################################################


## Valerie Pecresse (VP)


## create a .csv of VP's Twitter activity since 27 Jan 2022
!minet tw scrape tweets "(from:vpecresse) until:2022-03-27 since:2022-01-27" > tweets_VP.csv

## convert the .csv file in a data frame using pandas
df_tw_VP = pd.read_csv("./tweets_VP.csv")

## create a list of tweets selecting the 'text' column of the data frame
list_tw_VP = df_tw_VP['text'].values.tolist()
len(list_tw_VP)

## retrieve VP's affiliates1 Twitter activity
!minet tw scrape tweets "(from:MichelBarnier OR from:ChJacob77 OR from:ADublanche OR from:othmannasrou OR from:ECiotti OR from:xavierbertrand OR from:DebordValerie OR from:gerard_larcher) until:2022-03-27 since:2022-01-27" > tw_VP_aff_all1.csv

## convert VP's affiliates' tweets in a list
df_tw_VP_aff_all1 = pd.read_csv("tw_VP_aff_all1.csv")
list_tw_VP_aff_all1 = df_tw_VP_aff_all1['text'].values.tolist()
print(list_tw_VP_aff_all1[0])
print(len(list_tw_VP_aff_all1))

## retrieve VP's affiliates2 Twitter activity
!minet tw scrape tweets "(from:VincentJEANBRUN OR from:nadine__morano OR from:rezeg_hamida OR from:AurelienPradie OR from:CharlesConsigny OR from:GeoffroyDidier OR from:BrunoRetailleau OR from:GuilhemCarayon OR from:Herve_Morin) until:2022-03-24 since:2022-01-27" > tw_VP_aff_all2.csv

## convert VP's affiliates' tweets in a list
df_tw_VP_aff_all2 = pd.read_csv("tw_VP_aff_all2.csv")
list_tw_VP_aff_all2 = df_tw_VP_aff_all2['text'].values.tolist()
print(list_tw_VP_aff_all2[0])
print(len(list_tw_VP_aff_all2))

## merge the two lists of affiliates' tweets
list_tw_VP_aff_all = list_tw_VP_aff_all1 + list_tw_VP_aff_all2
print(len(list_tw_VP_aff_all))

## merge VP and his affiliates' lists
list_tw_VP_all = list_tw_VP + list_tw_VP_aff_all
print(len(list_tw_VP_all))

## i retrieve a string from the pdf of VP's manifesto using extract_text of the pdfminer package
## the cleaning process is specific for this manifesto and it depends on the output of extract_text
manif_VP = extract_text('/Users/simonemariaparazzoli/Documents/Università/Sciences Po/Diving into public digital spaces/research/manifesto_pecresse.pdf')
manif_clean_VP = manif_VP.replace('-\n','')
manif_clean_VP = manif_clean_VP.replace(' \n',' ')
manif_clean_VP = manif_clean_VP.replace('\n ','')
manif_clean_VP = manif_clean_VP.replace('\x0c',' ')
manif_clean_VP = manif_clean_VP.replace('\n\n','\n')
manif_clean_VP = manif_clean_VP.replace('\n','---')
#print(repr(manif_clean_VP))

## convert the string of the manifesto into a list
list_manif_VP = manif_clean_VP.split("---")
list_manif_VP = [s for s in list_manif_VP if len(s)>30]
len(list_manif_VP)
#print(list_manif_MLP)

## merge the two lists of tweets and of the manifesto 
list_VP = list_tw_VP_all + list_tw_VP_all + list_manif_VP
len(list_VP)

## load a spacy model to retrieve stop words
nlp = spacy.load("fr_core_news_sm")
stop_words_fr = nlp.Defaults.stop_words

## clean the list of tweets and manifesto to get rid of useless words and make the list content homogeneous
list_VP_clean = []
for i in list_VP:
    doc = nlp(i)
    tokens = [unidecode.unidecode(token.text).lower() for token in doc 
              if (token.text not in stop_words_fr and
                  len(token.text)>2 and
                  token.like_url == False )]
    tokens_joined = ' '.join(tokens)
    list_VP_clean.append(tokens_joined)
    
## test the output of the cleaning process
print(list_VP[205])
print("---")
print(list_VP_clean[205])

## prepare the corpus as a class
class MyCorpus_VP:

    def __iter__(self):
        for i in list_VP_clean:
            yield utils.simple_preprocess(i,min_len=3)
            
## train the word embeddings model_MLP
sentences = MyCorpus_VP()
model_VP = gensim.models.Word2Vec(sentences=sentences, min_count=10, vector_size=300, epochs=100)

## transform the corpus list (that is made of tweets and sentences from the manifesto)
## in a list containing all the words of the corpus as elements of the list
words_VP = []

for i in list_VP_clean:
    i_split = i.split(' ') #transform the i document into a list (split at blank space)
    words_VP.extend(i_split)

## clean the list of tokens
words_VP_clean = [x for x in words_VP
                   if x not in stop_words_fr
                   if x != "\n\n"
                   if len(x)>1]

## find the 30 most common words using Counter
words_freq_VP = Counter(words_VP_clean)
common_words_VP = words_freq_VP.most_common(30)
print(common_words_VP)

result = model_VP.wv.most_similar(positive=['france'], topn=20)
print(result)

# this is the old one! with no changes on epochs parameters
result = model_VP.wv.most_similar(positive=['france'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['etat'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['souverainete'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['president'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['politique'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['droit'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['entreprise'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['economie'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['emploi'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['travail'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['taxes'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['numerique'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['donnees'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['monde'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['realite'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['verite'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['societe'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['pouvoir'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['avenir'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['histoire'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['contre'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['faut'], topn=20)
print(result)

result = model_VP.wv.most_similar(positive=['crise'], topn=20)
print(result)


#########################################################
#########################################################


## Jean-Luc Mélenchon (JLM)


## create a .csv of JLM's Twitter activity since 27 Jan 2022
!minet tw scrape tweets "(from:JLMelenchon) until:2022-03-27 since:2022-01-27" > tweets_JLM.csv

## convert the .csv file in a data frame using pandas
df_tw_JLM = pd.read_csv("./tweets_JLM.csv")

## create a list of tweets selecting the 'text' column of the data frame
list_tw_JLM = df_tw_JLM['text'].values.tolist()
len(list_tw_JLM)

## retrieve JLM's affiliates1 Twitter activity
!minet tw scrape tweets "(from:AQuatennens OR from:JulieGarnierFI OR from:mbompard OR from:ALeaument OR from:Clemence_Guette OR from:Francois_Ruffin OR from:MathildePanot OR from:alexiscorbiere OR from:ClementVerde OR from:Aurelien_Le_Coq OR from:Deputee_Obono OR from:ericcoquerel OR from:Clem_Autain OR from:PrudhommeLoic OR from:BenedictTaurine) until:2022-03-27 since:2022-01-27" > tw_JLM_aff_all.csv

## convert JLM's affiliates' tweets in a list
df_tw_JLM_aff_all = pd.read_csv("tw_JLM_aff_all.csv")
list_tw_JLM_aff_all = df_tw_JLM_aff_all['text'].values.tolist()
print(list_tw_JLM_aff_all[0])
print(len(list_tw_JLM_aff_all))

## convert .csv affiliates' tweets in a list
df_tw_JLM_aff_all = pd.read_csv("tw_JLM_aff_all.csv")
list_tw_JLM_aff_all = df_tw_JLM_aff_all['text'].values.tolist()
print(list_tw_JLM_aff_all[0])
print(len(list_tw_JLM_aff_all))

## merge JLM's and JLM's affiliates lists
list_tw_JLM_all = list_tw_JLM + list_tw_JLM_aff_all

## i retrieve a string from the pdf of JLM's manifesto using extract_text of the pdfminer package
## the cleaning process is specific for this manifesto and it depends on the output of extract_text
manif_JLM = extract_text('/Users/simonemariaparazzoli/Documents/Università/Sciences Po/Diving into public digital spaces/research/manifesto_melenchon.pdf')
manif_clean_JLM = manif_JLM.replace(' .','')
manif_clean_JLM = manif_clean_JLM.replace('   ','')
manif_clean_JLM = manif_clean_JLM.replace('\n\n','---')
manif_clean_JLM = manif_clean_JLM.replace('\n','')
manif_clean_JLM = manif_clean_JLM.replace('\u202f',' ')
manif_clean_JLM = manif_clean_JLM.replace('\x0c',' ')
#print(repr(manif_clean_JLM))

## convert the string of the manifesto into a list
list_manif_JLM = manif_clean_JLM.split("---")
list_manif_JLM = [s for s in list_manif_JLM if len(s)>20]
len(list_manif_JLM)
#print(list_manif_EM)

## merge the two lists of tweets and of the manifesto 
list_JLM = list_tw_JLM_all + list_manif_JLM
len(list_JLM)

## load a spacy model to retrieve stop words
nlp = spacy.load("fr_core_news_sm")
stop_words_fr = nlp.Defaults.stop_words
#new_sw = ["avec","la","les","le","pour","un","une","nous","ete","et","je"]
#stop_words_fr.add(new_sw)

## clean the list of tweets and manifesto to get rid of useless words and make the list content homogeneous
list_JLM_clean = []
for i in list_JLM:
    doc = nlp(i)
    tokens = [unidecode.unidecode(token.text).lower()for token in doc 
              if ( token.text not in stop_words_fr and
                  len(token.text)>1 and
                  token.like_url == False )]
    tokens_joined = ' '.join(tokens)
    list_JLM_clean.append(tokens_joined)
    
## test the output of the cleaning process
print(list_JLM[2401])
print("---")
print(list_JLM_clean[2401])

## prepare the corpus as a class
class MyCorpus_JLM:

    def __iter__(self):
        for i in list_JLM_clean:
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(i,min_len=3)
            
## train the word embeddings model_JLM
sentences = MyCorpus_JLM()
model_JLM = gensim.models.Word2Vec(sentences=sentences, min_count=10, vector_size=300, epochs=100)

## transform the corpus list (that is made of tweets and sentences from the manifesto)
## in a list containing all the words of the corpus as elements of the list
words_JLM = []

for i in list_JLM_clean:
    i_split = i.split(' ') #transform the i document into a list (split at blank space)
    words_JLM.extend(i_split)

## clean the list of tokens
words_JLM_clean = [x for x in words_JLM 
                   if x not in stop_words_fr
                   if x != "\n\n"
                   if len(x)>1]

## find the 30 most common words using Counter
words_freq_JLM = Counter(words_JLM_clean)
common_words_JLM = words_freq_JLM.most_common(100)
print(common_words_JLM)

result = model_JLM.wv.most_similar(positive=['france'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['etat'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['souverainete'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['president'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['politique'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['droit'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['entreprise'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['economie'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['emploi'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['travail'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['numerique'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['donnees'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['monde'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['realite'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['verite'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['societe'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['pouvoir'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['avenir'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['histoire'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['contre'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['faut'], topn=20)
print(result)

result = model_JLM.wv.most_similar(positive=['crise'], topn=20)
print(result)


#########################################################
#########################################################


## Yannick Jadot (YJ)


## create a .csv of YJ's Twitter activity since 27 Jan 2022
!minet tw scrape tweets "(from:yjadot) until:2022-03-27 since:2022-01-27" > tweets_YJ.csv

## convert the .csv file in a data frame using pandas
df_tw_YJ = pd.read_csv("./tweets_YJ.csv")

## create a list of tweets selecting the 'text' column of the data frame
list_tw_YJ = df_tw_YJ['text'].values.tolist()
len(list_tw_YJ)

## retrieve YJ's affiliates1 Twitter activity
!minet tw scrape tweets "(from:_EvaSas OR from:julienbayou OR from:sandraregol OR from:iordanoff OR from:CoulombelAlain OR from:FThiollet OR from:marinetondelier OR from:Csagaspe OR from:delphinebatho OR from:EricPiolle OR from:hub_laferriere OR from:SabrinaSebaihi OR from:Melanie_Vogel_ OR from:MounirSatouri) until:2022-03-27 since:2022-01-27" > tw_YJ_aff_all.csv

## convert YJ's affiliates' tweets in a list
df_tw_YJ_aff_all = pd.read_csv("tw_YJ_aff_all.csv")
list_tw_YJ_aff_all = df_tw_YJ_aff_all['text'].values.tolist()
print(list_tw_YJ_aff_all[0])
print(len(list_tw_YJ_aff_all))

## convert .csv affiliates' tweets in a list
df_tw_YJ_aff_all = pd.read_csv("tw_YJ_aff_all.csv")
list_tw_YJ_aff_all = df_tw_YJ_aff_all['text'].values.tolist()
print(list_tw_YJ_aff_all[0])
print(len(list_tw_YJ_aff_all))

## merge YJ's and YJ's affiliates lists
list_tw_YJ_all = list_tw_YJ + list_tw_YJ_aff_all

## i retrieve a string from the pdf of YJ's manifesto using extract_text of the pdfminer package
## the cleaning process is specific for this manifesto and it depends on the output of extract_text
manif_YJ = extract_text('/Users/simonemariaparazzoli/Documents/Università/Sciences Po/Diving into public digital spaces/research/manifesto_jadot.pdf')
manif_clean_YJ = manif_YJ.replace(' .','')
manif_clean_YJ = manif_clean_YJ.replace('   ','')
manif_clean_YJ = manif_clean_YJ.replace('\n\n','')
manif_clean_YJ = manif_clean_YJ.replace('\n','')
manif_clean_YJ = manif_clean_YJ.replace('. ','---')
manif_clean_YJ = manif_clean_YJ.replace('\x0c',' ')
#print(repr(manif_clean_JLM))

## convert the string of the manifesto into a list
list_manif_YJ = manif_clean_YJ.split("---")
list_manif_YJ = [s for s in list_manif_YJ if len(s)>20]
len(list_manif_YJ)
#print(list_manif_EM)

## merge the two lists of tweets and of the manifesto 
list_YJ = list_tw_YJ_all + list_tw_YJ_all + list_manif_YJ
len(list_YJ)

## load a spacy model to retrieve stop words
nlp = spacy.load("fr_core_news_sm")
stop_words_fr = nlp.Defaults.stop_words
#new_sw = ["avec","la","les","le","pour","un","une","nous","ete","et","je"]
#stop_words_fr.add(new_sw)

## clean the list of tweets and manifesto to get rid of useless words and make the list content homogeneous
list_YJ_clean = []
for i in list_YJ:
    doc = nlp(i)
    tokens = [unidecode.unidecode(token.text).lower()for token in doc 
              if ( token.text not in stop_words_fr and
                  len(token.text)>1 and
                  token.like_url == False )]
    tokens_joined = ' '.join(tokens)
    list_YJ_clean.append(tokens_joined)
    
## test the output of the cleaning process
print(list_YJ[2401])
print("---")
print(list_YJ_clean[2401])

## prepare the corpus as a class
class MyCorpus_YJ:

    def __iter__(self):
        for i in list_YJ_clean:
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(i,min_len=3)

## train the word embeddings model_YJ
sentences = MyCorpus_YJ()
model_YJ = gensim.models.Word2Vec(sentences=sentences, min_count=10, vector_size=300, epochs=100)

## transform the corpus list (that is made of tweets and sentences from the manifesto)
## in a list containing all the words of the corpus as elements of the list
words_YJ = []

for i in list_YJ_clean:
    i_split = i.split(' ') #transform the i document into a list (split at blank space)
    words_YJ.extend(i_split)

## clean the list of tokens
words_YJ_clean = [x for x in words_YJ 
                   if x not in stop_words_fr
                   if x != "\n\n"
                   if len(x)>1]

## find the 30 most common words using Counter
words_freq_YJ = Counter(words_YJ_clean)
common_words_YJ = words_freq_YJ.most_common(30)
print(common_words_YJ)

result = model_YJ.wv.most_similar(positive=['france'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['etat'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['souverainete'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['president'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['politique'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['droit'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['entreprise'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['economie'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['emploi'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['travail'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['numerique'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['donnees'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['monde'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['realite'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['verite'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['societe'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['pouvoir'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['avenir'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['histoire'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['contre'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['faut'], topn=20)
print(result)

result = model_YJ.wv.most_similar(positive=['crise'], topn=20)
print(result)


#########################################################
#########################################################


## Anne Hidalgo (AH)


## create a .csv of AH's Twitter activity since 27 Jan 2022
!minet tw scrape tweets "(from:Anne_Hidalgo) until:2022-03-27 since:2022-01-27" > tweets_AH.csv

## convert the .csv file in a data frame using pandas
df_tw_AH = pd.read_csv("./tweets_AH.csv")

## create a list of tweets selecting the 'text' column of the data frame
list_tw_AH = df_tw_AH['text'].values.tolist()
len(list_tw_AH)

## retrieve AH's affiliates1 Twitter activity
!minet tw scrape tweets "(from:faureolivier OR from:Johanna_Rolland OR from:BorisVallaud OR from:Valerie_Rabault OR from:PatrickKanner OR from:RachidTemal OR from:RemiFeraud OR from:PJouvet OR from:SebVincini OR from:GabrielleSiry OR from:algrain_paris10 OR from:ACORDEBARD OR from:RemiFeraud OR from:PotierDominique) until:2022-03-27 since:2022-01-27" > tw_AH_aff_all.csv

## convert AH's affiliates' tweets in a list
df_tw_AH_aff_all = pd.read_csv("tw_AH_aff_all.csv")
list_tw_AH_aff_all = df_tw_AH_aff_all['text'].values.tolist()
print(list_tw_AH_aff_all[0])
print(len(list_tw_AH_aff_all))

## convert .csv affiliates' tweets in a list
df_tw_AH_aff_all = pd.read_csv("tw_AH_aff_all.csv")
list_tw_AH_aff_all = df_tw_AH_aff_all['text'].values.tolist()
print(list_tw_AH_aff_all[0])
print(len(list_tw_AH_aff_all))

## merge AH's and AH's affiliates lists
list_tw_AH_all = list_tw_AH + list_tw_AH_aff_all

## i retrieve a string from the pdf of AH's manifesto using extract_text of the pdfminer package
## the cleaning process is specific for this manifesto and it depends on the output of extract_text
manif_AH = extract_text('/Users/simonemariaparazzoli/Documents/Università/Sciences Po/Diving into public digital spaces/research/manifesto_hidalgo.pdf')
manif_clean_AH = manif_AH.replace(' .','')
manif_clean_AH = manif_clean_AH.replace('   ','')
manif_clean_AH = manif_clean_AH.replace('\n\n','')
manif_clean_AH = manif_clean_AH.replace('\n','')
manif_clean_AH = manif_clean_AH.replace('. ','---')
manif_clean_AH = manif_clean_AH.replace(' _ ','---')
manif_clean_AH = manif_clean_AH.replace('\x0c',' ')
#print(repr(manif_clean_AH))

## convert the string of the manifesto into a list
list_manif_AH = manif_clean_AH.split("---")
list_manif_AH = [s for s in list_manif_AH if len(s)>20]
len(list_manif_AH)
#print(list_manif_EM)

## merge the two lists of tweets and of the manifesto 
list_AH = list_tw_AH_all + list_tw_AH_all + list_manif_AH
len(list_AH)

## load a spacy model to retrieve stop words
nlp = spacy.load("fr_core_news_sm")
stop_words_fr = nlp.Defaults.stop_words
#new_sw = ["avec","la","les","le","pour","un","une","nous","ete","et","je"]
#stop_words_fr.add(new_sw)

## clean the list of tweets and manifesto to get rid of useless words and make the list content homogeneous
list_AH_clean = []
for i in list_AH:
    doc = nlp(i)
    tokens = [unidecode.unidecode(token.text).lower()for token in doc 
              if ( token.text not in stop_words_fr and
                  len(token.text)>1 and
                  token.like_url == False )]
    tokens_joined = ' '.join(tokens)
    list_AH_clean.append(tokens_joined)
    
## test the output of the cleaning process
print(list_AH[2401])
print("---")
print(list_AH_clean[2401])

## prepare the corpus as a class
class MyCorpus_AH:

    def __iter__(self):
        for i in list_AH_clean:
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(i,min_len=3)
            
## train the word embeddings model_AH
sentences = MyCorpus_AH()
model_AH = gensim.models.Word2Vec(sentences=sentences, min_count=10, vector_size=300, epochs=100)

## transform the corpus list (that is made of tweets and sentences from the manifesto)
## in a list containing all the words of the corpus as elements of the list
words_AH = []

for i in list_AH_clean:
    i_split = i.split(' ') #transform the i document into a list (split at blank space)
    words_AH.extend(i_split)

## clean the list of tokens
words_AH_clean = [x for x in words_AH 
                   if x not in stop_words_fr
                   if x != "\n\n"
                   if len(x)>1]

## find the 30 most common words using Counter
words_freq_AH = Counter(words_AH_clean)
common_words_AH = words_freq_AH.most_common(30)
print(common_words_AH)

result = model_AH.wv.most_similar(positive=['france'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['etat'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['souverainete'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['president'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['politique'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['droit'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['entreprise'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['economie'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['emploi'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['travail'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['numerique'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['donnees'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['monde'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['realite'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['verite'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['societe'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['pouvoir'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['avenir'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['histoire'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['contre'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['faut'], topn=20)
print(result)

result = model_AH.wv.most_similar(positive=['crise'], topn=20)
print(result)


#########################################################
#########################################################
