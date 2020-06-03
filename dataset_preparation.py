import fitz
import numpy as np
import re, string
from gensim import corpora, models
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

valid = words.words()
wnl = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = stopwords.words('english')

class myBook:
  def __init__(self, book_name):
    self.doc = fitz.open(book_name)

  def read_topics(self):
    #doc = file.open(self.name)
    a = self.doc.getToC()
    list_topic = []
    for i in a:
        list_topic.append(i[1])
    list_topic = list_topic[4:]
    
    count=0
    for i in range(0,len(list_topic)):
        list_topic[i] = list_topic[i].replace('\r',' ')
        if(list_topic[i]=='Exercises'):
            count+=1
    #print(count)
    for i in range(13):
        list_topic.remove('Exercises');
    #print(list_topic)
    return list_topic

  def book_to_list(self):
    text = []
    for page_number in range(21,self.doc.pageCount):
        text += self.doc.loadPage(page_number).getText('text').split('\n')[2:]
    # print(doc.loadPage(25).getText('text').split('\n')[2:])
    print(text)
    return text

  def get_index_of_topics(self, list_topic, text):
    index_topics = []
    count=0
    for i in list_topic:
        try:
            index_topics += [[i,text.index(i)]]
        except ValueError:
            try:
                if(len(i)>4 and i[5]==' '):
                    count+=1
                    string=i.split()[0]
                    index_topics +=[[i,text.index(str(string))+1]]
                if(len(i)>4 and i[6]==' '):
                    count+=1
                    string=i.split()[0]
                    index_topics +=[[i,text.index(str(string))+1]]
            except ValueError:
                # print('.'+i[5]+'.'+i[6:])
                continue
            continue
    # print(index_topics)
    return index_topics

  def map_topic_to_data(self, index_topics, text):
    topics_dict ={}
    for i in range(len(index_topics)-1):
        temp = ''
        temp_text = text[index_topics[i][1]:index_topics[i+1][1]]
        for j in temp_text:
            temp = temp+' '+j
        topics_dict[index_topics[i][0]] = temp
    #print(topics_dict)
    return topics_dict

  def preprocess_data(self, text):
    text = re.sub(r'[^\w\s]','',text.lower())
    text = re.sub('[^A-Za-z]', ' ', text)
    text = re.sub(r'\b\w{1,3}\b', '', text)
    tokenized_text = word_tokenize(text)
    clean_text = [wnl.lemmatize(word, 'n') for word in tokenized_text if word not in stop_words]
    return clean_text

  def preprocess_topics(self,lda_dataset):
    temp = list(lda_dataset.keys())
    for i in range(len(temp)):
      temp[i] = re.sub('[^A-Za-z]', ' ', temp[i].lower()).split()
      temp[i] = [word for word in temp[i] if word not in stop_words]
      print(temp[i])
    return temp

  def save_dataset(self, topics, lda_dataset):
    temp = list(lda_dataset.keys())
    file = open('lda_dataset_bishop_xx.txt','w+')
    for i in range(len(topics)):
        a1=''
        for j in lda_dataset[temp[i]]:
            a1+=' '+j
        file.write((str(topics[i])+a1+'\n'))
    file.close()


book_name = 'bishop.pdf'
book = myBook(book_name)
list_topics = book.read_topics()
print(list_topics)
text = book.book_to_list()
print(text)
index_topics  = book.get_index_of_topics(list_topics, text)
print(index_topics)
topics_dict = book.map_topic_to_data(index_topics, text)
print(topics_dict)

lda_dataset = {}
for i in topics_dict.keys():
    lda_dataset[i] = book.preprocess_data(topics_dict[i])
    lda_dataset[i] = [word for word in lda_dataset[i] if word in valid]
topics = book.preprocess_topics(lda_dataset)

book.save_dataset(topics, lda_dataset)
