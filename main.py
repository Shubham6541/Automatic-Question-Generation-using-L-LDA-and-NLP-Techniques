import tomotopy as tp
import numpy as np
import pandas as pd
import wikipedia
import matplotlib.pyplot as plt
from googlesearch import search
import pickle,re,string
from collections import defaultdict, Counter
from gensim import corpora, models
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from scipy import spatial

stop_words = stopwords.words('english')
w2v = pickle.load(open('Cleaned_W2V.pkl','rb'))
lem = WordNetLemmatizer()
stemmer = PorterStemmer()

class myLLDA():
  def __init__(self):
    self.mdl = tp.LLDAModel(tw=tp.TermWeight.IDF)

  def train_llda(self,dataset):
    for line in open(dataset):
        words = line.split(']')[1]
        label = line.split(']')[0][1:]
        label = label.split("'")[1::2]
        '''
        l = ''
        for i in label.split("'")[1::2]:
            l += i+'_'
        label = list()
        label.append(str(l[:-1]))
        '''
        for i in range(len(label)):
            label[i] = lem.lemmatize(label[i],pos='n')
            
        self.mdl.add_doc(words.split(),label)
        #print(label)

    for i in range(0, 100, 100):
        self.mdl.train(100)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, self.mdl.ll_per_word))

    for k in range(self.mdl.k):
        print('Top 10 words of topic #{}'.format(k))
        print(self.mdl.get_topic_words(k, top_n=10))
    print(len(self.mdl.topic_label_dict))
    print(self.mdl.topic_label_dict)

  def training_evaluation(self, dataset):
    training_acc = {} 
    training_acc_sum = {}
    for line in open(dataset,'r'):
      training_words = line.split(']')[1]     
      label = line.split(']')[0][1:]     
      label_list = label.split("'")[1::2]
      for i in range(len(label_list)):
            label_list[i] = lem.lemmatize(label_list[i],pos='n')     
      doc_inst = self.mdl.make_doc(training_words.split())
      topic_dist, ll = self.mdl.infer(doc_inst)
      try:

        training_acc[label] = [topic_dist[np.where(np.array(self.mdl.topic_label_dict)==i)[0][0]] for i in label_list]
        training_acc_sum[label] = sum(training_acc[label])  
      except IndexError:
        print(label)

    print(training_acc)
    print(training_acc_sum)
    print(np.mean(list(training_acc_sum.values())))

  def preprocess_data(self, text):
    text = re.sub(r'[^\w\s]','',text.lower())
    text = re.sub('[^A-Za-z]', ' ', text)
    text = re.sub(r'\b\w{1,3}\b', '', text)
    tokenized_text = word_tokenize(text)
    clean_text = [lem.lemmatize(word, 'n') for word in tokenized_text if word not in stop_words]
    return clean_text

  def testing_llda(self, test_data):
    doc_inst = self.mdl.make_doc(test_data)
    topic_dist, ll = self.mdl.infer(doc_inst)
    print("Topic Distribution for Unseen Docs: ", max(topic_dist))
    print("Log-likelihood of inference: ", ll)
    l1 = topic_dist[:]
    topic = []
    topic_value = []
    
    for i in range(10):
        print(self.mdl.topic_label_dict[np.argmax(l1)],np.max(l1))
        topic.append(self.mdl.topic_label_dict[np.argmax(l1)])
        topic_value.append(np.max(l1))
        l1 [np.argmax(l1)] = -100
      
    font  = {"family":"sans-serif","color":"black","weight":"normal","size":16}

    plt.xlabel('Topics',fontdict =font) 
    plt.ylabel('Probabilty',fontdict =font)
    plt.bar(topic[:5],topic_value[:5]) 
    plt.title("Unweighted Probability vs Topic",fontdict =font)
    plt.show()
    return topic_dist

  def testing_llda_weighted(self, topic_dict,test_data):
    distribution_list = topic_dict[:]
    list_of_topics = list(self.mdl.topic_label_dict)
    weights = defaultdict(lambda: 0)
    count_inDocument = defaultdict(lambda: 0)
    for i in range(len(list_of_topics)):
        weights[list_of_topics[i]] += distribution_list[i]
      
    for i in range(len(test_data)):
        if (test_data[i] in list_of_topics and weights[test_data[i]] < 0.5):
            weights[test_data[i]] += 0.06/(2**count_inDocument[test_data[i]])
            count_inDocument[test_data[i]] += 1
    
    k = Counter(weights)
    high = k.most_common(5)
    print(high,type(high))
    sum_top_five = 0
    for i in high:
      sum_top_five += i[1]
    for i in high: 
        print(i[0]," :",i[1]," ")
    font  = {"family":"sans-serif","color":"black","weight":"normal","size":16}
    top_five = np.array(high) 
    x = top_five[:5,0] 
    y = np.array(top_five[:5,1],dtype=float)/sum_top_five
    print(y)
    plt.xlabel('Topics',fontdict =font) 
    plt.ylabel('Probabilty',fontdict =font)
    plt.bar(x,y) 
    plt.title("Weighted Probability vs Topic",fontdict =font)
    plt.show() 
    return weights,high

  def find_questions(self, top_topics,ngrams=1):
    if(ngrams == 1):
      h = np.array(top_topics)
      h = h[:,0]
      print(h)
      questions = []
      for j in range(0,len(h)):
          print(str((j+1))+'.'+h[j])
          for i in search('explain '+h[j]+' quora',tld='com' ,stop=3,pause=2):
              #print(i)
              i = i.split('/')[-1]
              i = i.replace('-',' ')
              print('   Q.'+' '+i+'?')
              questions.append(i)
      return questions
    if(ngrams == 3):
      for l in range(len(h)-2):
        for j in range(l+1,len(h)-1):
          for k in range(j+1,len(h)):
             index_ += 1
             print(str((index_))+'.'+h[l] +' '+ h[j]+' '+h[k])
             for i in search('explain '+h[l] +' '+ h[j]+' '+h[k]+' quora',tld='com' ,stop=3,pause=2):
               # print(i)
               i = i.split('/')[-1]
               i = i.replace('-',' ')
               print('   Q.'+' '+i+'?')
               questions.append(i)
      return questions

    if(ngrams == 5):
      index_ = 0
      questions = []
      #print(h[0]+' '+h[1]+' '+h[2]+' '+h[3]+' '+h[4])
      for i in search('explain '+h[0]+' '+h[1]+' '+h[2]+' '+h[3]+' '+h[4]+' quora',tld='com' ,stop=5,pause=2):
        #print(i)
        i = i.split('/')[-1]
        i = i.replace('-',' ')
        print('   Q.'+' '+i+'?')
        questions.append(i)
      return questions

  def cosine_similarity(self, a1, a2):
    return 1 - spatial.distance.cosine(a1, a2)

  def top_questions(self, test_data, questions):
    similarity = -10000
    question = defaultdict(lambda:'asd')
    v1 = []
    question_df ={}
    for i in test_data:
        try:
            v1.append(w2v[i])
        except KeyError:
            continue

    v2 = []
    for I in questions:
        temp = []
        i = self.preprocess_data(I)
        for j in i: 
            try:
                v2.append(w2v[j])
            except KeyError:
                continue
        for v_question in v2:
          temp2 = 0
          for v_data in v1:
            temp2+= self.cosine_similarity(v_data, v_question)
          temp.append(temp2)
        try:
          print(temp)
          length = len(temp)
          temp_max = max(temp)
          temp_avg = np.mean(temp)
        except ValueError:
          temp = 0
        question[I] = temp_max
        
        v2 = []
        question_df[I] = [temp_max/len(test_data),temp_avg/len(test_data)]
    return question, question_df 

model = myLLDA()
dataset = 'lda_dataset_bishop_xx.txt'

model.train_llda(dataset)
model.training_evaluation(dataset)

test_data = 'Feedforward neural network'
test_data = (wikipedia.page(test_data).content)
test_data = model.preprocess_data(test_data)
topic_dict = model.testing_llda(test_data)
weighted_topic_dict, top_topics = model.testing_llda_weighted(topic_dict,test_data)
questions = model.find_questions(top_topics)
question, question_df = model.top_questions(test_data,questions)

pd.set_option('max_colwidth',-1)
df = pd.DataFrame.from_dict(question_df,orient='index')
k = Counter(question)
top_questions = k.most_common(5)
for i in top_questions: 
  print(np.round(i[1]/len(test_data),3)," :",i[0]," ")

df = df.rename(columns={0:'max distribution',1:'avg distribution'})
df = df.sort_values(by='max distribution',ascending=False)
df.to_csv('best_questions.csv')
