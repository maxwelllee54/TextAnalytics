#by Jenny GONG
import nltk 
from nltk.corpus import stopwords
from nltk import FreqDist

f = open('input_new.txt','r')
raw = f.read()
raw = raw.replace('\n',' ')

#Tokenization
tokens = nltk.word_tokenize(raw)

#Stopwords Removal and only keep text data then change to lowercase
mystopwords = stopwords.words('english')
words = [w.lower() for w in tokens if w.isalpha() if w.lower()not in mystopwords]


#### You can change into different stemmer, but we use Porter Stemmer here

#Use Porter Stemmer 
porter = nltk.PorterStemmer()
stem1 = [porter.stem(w) for w in words]
#Encode with utf-8
stem1 = [w.encode('utf8') for w in stem1]
#Get the frequency distribution 
freq1 = FreqDist(stem1)
#Sort the result
sorted_freq1 = sorted(freq1.items(),key = lambda k: k[1], reverse = True)


#write result into .txt file
with open('stemming_output.txt','w') as f:
    for word, frequency in sorted_freq1: #here you can change to sorted_freq2 or 3 
        f.write(str(word)+'\t'+str(frequency)+'\n')

        