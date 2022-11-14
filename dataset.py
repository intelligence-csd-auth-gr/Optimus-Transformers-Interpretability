from nltk import TreebankWordTokenizer
import pandas as pd
import numpy as np
from scipy import stats
import urllib
import re
import matplotlib.pyplot as plt
import pickle
import json
from transformers import BertTokenizerFast

def my_clean(text):
	""" This function applies a set of cleanup procedures to a text
    Args:
		text: The input text to be cleaned
    """
	text = str(text)
	text = re.sub(r"[^A-Za-z0-9^,!\/'+-=]", " ", text)
	text = re.sub(r",", " ", text)
	text = re.sub(r"\d\.", "", text)
	text = re.sub(r"!", " ! ", text)
	text = re.sub(r"\/", " ", text)
	text = re.sub(r"\.", " .", text)
	text = re.sub(r"\^", " ^ ", text)
	text = re.sub(r"\+", " + ", text)
	text = re.sub(r" \- ", " ", text)
	text = re.sub(r"\- ", " ", text)
	text = re.sub(r" \-", " ", text)
	text = re.sub(r"\-", " ", text)
	text = re.sub(r"\r", "", text)
	text = re.sub(r"\n", "", text)
	text = re.sub(r"\=", " = ", text)
	text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	text = re.sub(r":", " ", text)
	text = re.sub(r";", " ", text)
	text = re.sub(r"\s{2,}", " ", text)
	if text.split(' ')[0] == '':
		return ' '.join(text.split(' ')[1:])
	return text

def hoc_preprocess(input_text):
	""" This function applies a set of cleanup procedures to texts of HOC dataset specifically
    Args:
		text: The input text to be cleaned
    """
	text = input_text.lower()
	text = text.encode().decode('unicode_escape')
	print(text)
	text = re.sub(r'[^\x00-\x7F]+', r'', text)
	text = re.sub(r"\s's\b", r"'s", text)
	text = re.sub(r'(\S)\.', r'\g<1>,', text)
	text = re.sub(r'\.(\S)', r',\g<1>', text)
	text = re.sub(r'\-', r' ', text)
	text = text.decode()
	print(text)
	return text

class Dataset:
	""" This class contains functions to load the datasets we used in our paper
    """
	def __init__(self, path=None, x=None, y=None, rationales=None ,label_names=None):
		""" This initiates instances of this class
    	Args:
			path: the path to load the datasets
			x: This contains the instances
			y: This contains the labels
			label_names: The labels names
			rationales: The ground truth interpretations
    	"""
		self.path = path
		self.x = x
		self.y = y
		self.label_names = label_names
		self.rationales =rationales

	def load_movies(self,level= 'token'):
		""" This loads the movies dataset: DeYoung, Jay, et al. "ERASER: A benchmark to evaluate rationalized NLP models." arXiv preprint arXiv:1911.03429 (2019).
    	Args:
			level: if it loads the rationales at token or sentence level
    	"""
		path = 'datasets/'
		with open(path+'movies.pickle', 'rb') as handle:
			dataset = pickle.load(handle)
		text = []
		labels = []
		rationales=[]
		for key in dataset.keys():
			text.append(dataset[key][0])
			if dataset[key][3]['classification'] == 'NEG':
				labels.append(0)
			else:
				labels.append(1)
			if level == 'sentence':
				rationales.append(dataset[key][2])
			else:
				rationales.append(dataset[key][1])
		self.x = text
		self.y = np.array(labels)
		self.rationales = np.array(rationales)
		self.label_names = ['NEG','POS']
		return self.x, self.y, self.label_names, self.rationales

	def load_esnli(self):
		""" This loads the ESNLI dataset: Camburu, Oana-Maria, et al. "e-snli: Natural language inference with natural language explanations." Advances in Neural Information Processing Systems 31 (2018).
    	"""
		path = 'datasets/'
		with open(path+'esnli.pickle', 'rb') as handle:
			self.dataset = pickle.load(handle)
		self.label_names = ['contradiction','entailment']
		return self.dataset, self.label_names 

	def load_hummingbird(self):
		""" This loads the hummingbird dataset: Hayati, Shirley, Dongyeop Kang, and Lyle Ungar. "Does BERT Learn as Humans Perceive? Understanding Linguistic Styles through Lexica." Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 2021.
    	"""
		path = 'datasets/hummingbird/'
		dfs = []
		for file in ['anger.tsv', 'disgust.tsv', 'fear.tsv', 'joy.tsv', 'offensive.tsv', 'sadness.tsv']:
			dfs.append(pd.read_csv(path+file, delimiter='\t'))
		X = dfs[0]['processed_text'].values

		y = [dfs[0]['human_label'].values]
		for df in dfs[1:]:
			y.append(df['human_label'].values)
		y = np.array(y).T
		
		z = [[] for i in range(500)]
		for df in dfs:
			counter = 0
			for ps in df['perception_scores']:
				z[counter].append([float(i) for i in (ps.split(' '))])
				counter = counter + 1
		z = np.array(z)
		self.x = X
		self.y = 1 - y
		self.rationales = z
		self.label_names = ['anger', 'disgust', 'fear', 'joy', 'offensive', 'sadness']
		return self.x, self.y, self.label_names, self.rationales

	def load_AIS(self, preprocess=False, plot=False):
		""" This loads the AIS (acute ischemic stroke) dataset: Kim, Chulho, et al. "Natural language processing and machine learning algorithm to identify brain MRI reports with acute ischemic stroke." PloS one 14.2 (2019): e0212778.
    	Args:
			preprocess: if it preprocess the text
			plot: if it plots the statistics of the dataset (number of tokens, etc.)
    	"""
		self.label_names = ['Bob', 'Patrick']
		data = pd.read_csv(self.path+'datasets/AIS.csv')
		X = list(data['Text'].values)
		if preprocess:
			X = [my_clean(x) for x in X]
		y = data['Label'].values
		if plot:
			mmm = 0
			sss = 0
			bbb = []
			for x in X:
				lll = len(x.split(' '))
				mmm = max(mmm, lll)
				sss = sss + lll
				bbb.append(lll)
			mmm, sss/len(X)
			n, bins, patches = plt.hist(bbb, 50, density=True, facecolor='g', alpha=0.75)
			plt.title('Before sequence cut')
			plt.show()
		x_new = []
		for x in X:
			splits = x.split(' ')
			if len(splits) >250:
				x_new.append(' '.join(x.split(' ')[:250]))
			else:
				x_new.append(' '.join(x.split(' ')))
		if plot:
			mmm = 0
			sss = 0
			bbb = []
			for x in x_new:
				lll = len(x.split(' '))
				mmm = max(mmm, lll)
				sss = sss + lll
				bbb.append(lll)
			mmm, sss/len(x_new)
			n, bins, patches = plt.hist(bbb, 50, density=True, facecolor='g', alpha=0.75)
			plt.title('After sequence cut')
			plt.show()
		self.x = x_new
		self.y = y
		return self.x, self.y, self.label_names

	def load_ethos(self, preprocess=False):
		""" This loads the ethos dataset. Mollas, Ioannis, et al. "ETHOS: a multi-label hate speech detection dataset." Complex & Intelligent Systems (2022): 1-16.
    	Args:
			preprocess: if it preprocess the text
    	"""
		self.label_names = ['hate speech','violence','directed_vs_generalized','gender','race','national_origin','disability','religion','sexual_orientation']
		data = pd.read_csv(
			self.path+"datasets/ethos/binary.csv", delimiter=';')
		
		np.random.seed(2000)
		data = data.iloc[np.random.permutation(len(data))]
		XT = data['comment'].values
		X = []
		yT = data['isHate'].values
		y = []
		count = 0
		for i in range(len(yT)):
			if yT[i] < 0.5:
				y.append([0,0,0,0,0,0,0,0,0])
				X.append(XT[i])
		
		data = pd.read_csv(
			self.path+"datasets/ethos/multilabel.csv", delimiter=';')
		X = []
		y = []
		XT = data['comment'].values
		yT = data.loc[:, data.columns != 'comment'].values
		for j in range(len(yT)):
			#yi = [1]
			yi = []
			yt = yT[j]
			for i in yt:
				if i >= 0.5:
					yi.append(int(1))
				else:
					yi.append(int(0))
			y.append(yi)
			X.append(XT[j])
		if preprocess:
			X = [my_clean(x) for x in X]
		
		self.x = X
		self.y = y
		return self.x, self.y, self.label_names

	def load_hoc(self, preprocess=False):
		""" This loads the hoc (hallmarks of cancer) dataset: Baker, Simon, et al. "Automatic semantic classification of scientific literature according to the hallmarks of cancer." Bioinformatics 32.3 (2016): 432-440.
    	Args:
			preprocess: if it preprocess the text
    	"""
		with open(self.path+'datasets/hoc_preprocessed.pkl', 'rb') as handle:
			hoc_dict = pickle.load(handle)

		X= hoc_dict['x']
		x_new = []
		if preprocess:
			 X = [my_clean(x) for x in X]
		#X = [hoc_preprocess(x) for x in X]
		for x in X:
			splits = x.split(' ')
			if len(splits) > 300:
				temp = ' '.join(x.split(' ')[:300])
				if temp[-1] == '.':
					x_new.append(temp+' ')
				elif temp[-1] == ' ' and temp[-2] == '.':
					x_new.append(temp)
				else:	
					x_new.append(' '.join(x.split(' ')[:300])+' .')
			else:
				x_new.append(' '.join(x.split(' ')))
		y_binarized = hoc_dict['y_binarized']
		z= hoc_dict['z']
		label_names = hoc_dict['label_names']
		self.x = x_new
		self.y = y_binarized
		self.rationales = z
		self.label_names = label_names
		return self.x, self.y, self.label_names,  self.rationales
	

	def load_hatexplain(self, tokenizer, preprocess=False, plot=False):
		""" This loads the hatexplain dataset. Mathew, Binny, et al. "Hatexplain: A benchmark dataset for explainable hate speech detection." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 17. 2021.
    	Args:
			tokenizer: the tokenizer to be used for spliting the input sequence and assign the rational scores
			preprocess: if it preprocess the text
			plot: if it plots the statistics of the dataset (number of tokens, etc.)
    	"""
		def getGroundTruth(key, tokens):
			original_rationales = data[key]['rationales']
			new_rationales = []
			lengths = []
			for token in tokens:
				lengths.append(len(tokenizer.tokenize(token)))
			for current_rationale in original_rationales:
				tweaked_rationale = []
				for weight, length in zip(current_rationale, lengths):
					tweaked_rationale += length * [weight]
				new_rationales.append(tweaked_rationale)

			ground_truth = [int(any(weight)) for weight in zip(*new_rationales)]
			return ground_truth

		def preprocess(tokens):
			for i in range(len(tokens)):
				tokens[i] = re.sub(r"[<\*>]", "", tokens[i])
				tokens[i] = re.sub(r"\b\d+\b", "number", tokens[i])
			return tokens

		with open(self.path+'datasets/hatexplain.json', 'r') as fp:
				data = json.load(fp)
		X = []
		y = []
		ground_truth = []
		for key in data:
			tokens = data[key]['post_tokens']
			tokens = preprocess(tokens)
			text = ' '.join(tokens)
			annotator_labels = []
			for i in range(3):
					annotator_labels.append(data[key]['annotators'][i]['label'])
		
			final_label = max(annotator_labels, key=annotator_labels.count)
		
			if (annotator_labels.count(final_label) != 1):
				if (final_label == 'hatespeech'):
					X.append(text)
					y.append(int(1))
					ground_truth.append(getGroundTruth(key, tokens))
				elif (final_label == 'normal'):
					X.append(text)
					y.append(int(0))
					ground_truth.append(int(0))
		
		self.x = X
		self.y = y
		self.rationales = ground_truth
		self.label_names = ['noHateSpeech', 'hateSpeech']
		
		return self.x , self.y , self.label_names, self.rationales