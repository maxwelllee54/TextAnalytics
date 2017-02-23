#by Jenny GONG
import nltk 
from nltk.corpus import stopwords
from nltk import FreqDist

f = open('input_new.txt','r')
raw = f.read()
raw = raw.replace('\n',' ')

#Tokenization
tokens = nltk.word_tokenize(raw)


#POS Tagging
POS_tags = nltk.pos_tag(tokens) #use unprocessed 'tokens', not 'words'

#Generate a list of POS tags
POS_tag_list = [(word,tag) for (word,tag) in POS_tags if tag.startswith('N')]

#Generate a frequency distribution of all the POS tags
tag_freq = nltk.FreqDist(POS_tag_list)
#Sort the result 
sorted_tag_freq = sorted(tag_freq.items(), key = lambda k:k[1], reverse = True)

#write result into .txt file
with open('POS_output.txt','w') as f:
    for (word,tag),frequency in sorted_tag_freq:
        f.write(str(word)+'\t'+str(tag)+'\t'+str(frequency)+'\n')