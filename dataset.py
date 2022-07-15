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

from nltk import TreebankWordTokenizer
def hoc_preprocess(input_text):
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
	def __init__(self, path=None, x=None, y=None, rationales=None ,label_names=None):
		self.path = path
		self.x = x
		self.y = y
		self.label_names = label_names
		self.rationales =rationales
		
	def load_AIS(self, preprocess=False, plot=False):
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