#by Jenny GONG
import nltk
from nltk import FreqDist

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

#encode the raw token list by utf-8
words3 = [w.encode('utf8') for w in words2]

#generate a frequency dictionary for all tokens 
freq = FreqDist(words3)

#sort the frequency list in decending order
sorted_freq = sorted(freq.items(),key = lambda k:k[1], reverse = True)

with open ('output.txt','a') as outfile:
    for line in sorted_freq:
        outfile.write(str(line[0])+'\t'+str(line[1])+'\n')
