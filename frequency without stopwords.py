#by Jenny GONG
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords

#read file from local 
f = open('input.txt','rU')
raw = f.read()

raw = raw.replace('\n',' ') 
raw = raw.decode('utf8') #decode raw text by utf-8

tokens = nltk.word_tokenize(raw)
#change all tokens into lower case 
words1 = [w.lower() for w in tokens]   #list comprehension 
#only keep text words, no numbers 
words2 = [w for w in words1 if w.isalpha()]

stopwords = stopwords.words('english') #use the NLTK stopwords

#only keep the words that not in nltk stopwords word list
words_nostopwords = [w.encode('utf8') for w in words2 if w not in stopwords]
#generate a frequency dictionary for all tokens 
freq_nostw = FreqDist(words_nostopwords)
#sort the frequency list in decending order
sorted_freq_nostw = sorted(freq_nostw.items(),key = lambda k:k[1], reverse = True)



with open ('output_nostopwords.txt','a') as outfile:
    for line in sorted_freq_nostw:
        outfile.write(str(line[0])+'\t'+str(line[1])+'\n')

