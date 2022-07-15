from scipy.special import softmax
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

class MyEvaluation:
	def __init__(self, label_names, predict, sentence_level):
		self.label_names = label_names
		self.predict = predict
		self.sentence_level = sentence_level
		self.saved_state = {}

	def clear_states(self):
		"""
			This function resets the state memory used by thruthfulness metric
		"""
		self.saved_state = {}

	def fix_instance(self, instance):
		"""
			This function takes an instance like "The tok ##en ##ized sentence" and transforms it to "The tokenized sentence"
		"""
		new_sentence = ''
		temp_split = instance.split()
		for i in range(0,len(temp_split)):
			if "##" not in temp_split[i]:
				new_sentence = new_sentence + ' ' + temp_split[i]
			else:
				new_sentence = new_sentence + temp_split[i][2:]
		return new_sentence[1:]

	def _find_sign(self, weight):
		if weight < 0:
			sign = 'negative'
		elif weight > 0:
			sign = 'positive'
		else:
			sign = 'neutral'
		return sign

	def _apply_activation(self, pred1, pred2):
		if len(self.label_names) == 2:
			predicted_labels = softmax([pred1,pred2], axis = 1)
		else:
			predicted_labels = [pred1,pred2]
			a = tf.constant(predicted_labels, dtype = tf.float32)
			b = tf.keras.activations.sigmoid(a)
			predicted_labels = b.numpy()
		return predicted_labels

	def nzw(self, interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales):
		av_nzw = []
		predicted_labels = self._apply_activation(prediction, prediction)[0]

		if(self.sentence_level == True):
			for label in range(len(self.label_names)):
				if predicted_labels[label]>=0.5:
					non_zero = 0
					threshold = 0.01
					for i in range(len(interpretation[label])):  
						if abs(interpretation[label][i]) > threshold:
							non_zero += 1
					av_nzw.append(non_zero/len(tokens))
				else:
					av_nzw.append(np.average([]))
			return av_nzw

		for label in range(len(self.label_names)):
			if predicted_labels[label]>=0.5:
				non_zero = 0
				threshold = 0.01
				for i in range(len(interpretation[label])):  
					if abs(interpretation[label][i]) > threshold:
						non_zero += 1
				av_nzw.append(non_zero/len(tokens[:-2])) # -2 to exclude cls and sep
			else:
				av_nzw.append(np.average([]))
		return av_nzw

	def faithfulness(self, interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales):
		avg_diff = []
		predicted_labels = self._apply_activation(prediction, prediction)[0]

		for label in range(len(self.label_names)):

			if predicted_labels[label]>=0.5:

				#print('For label:',self.label_names[label])
				absmax_index = np.argmax(interpretation[label])
				sign = self._find_sign(interpretation[label][absmax_index])
				if sign == 'negative':
					absmax_index = np.argmax([abs(i) for i in interpretation[label]])
					sign = self._find_sign(interpretation[label][absmax_index])
			
				temp_tokens = tokens.copy()
				#print('Argmax:',interpretation[label][absmax_index],'token:',temp_tokens[absmax_index+1])
				if self.sentence_level:
					temp_tokens[absmax_index] = ''
					temp_instance = ' '.join(temp_tokens)
				else:
					temp_tokens[absmax_index+1] = '[UNK]' # add '' to test the classic removal process
					temp_instance = self.fix_instance(' '.join(temp_tokens[1:-1]))
				if temp_instance in self.saved_state:
					temp_prediction = self.saved_state[temp_instance]
				else:
					temp_prediction, _, _ = self.predict(temp_instance)
					self.saved_state[temp_instance] = temp_prediction
				preds = self._apply_activation(prediction, temp_prediction)
				if sign == 'positive':
					diff = preds[0][label] - preds[1][label]
				elif sign == 'negative':
					diff = preds[1][label] -  preds[0][label]
				else: #neutral
					diff = (-1)*abs(preds[1][label] -  preds[0][label]) #Penalty
				avg_diff.append(diff)
			else:
				avg_diff.append(np.average([]))
		return avg_diff

	def truthfulness(self, interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales):
		avg_diff = []
		predicted_labels = self._apply_activation(prediction, prediction)[0]

		for label in range(len(self.label_names)):

			if predicted_labels[label]>=0.5:
				truthful = 0
				my_range = len(tokens) if self.sentence_level else len(tokens)-2
				#print('For label:',self.label_names[label])
				for token in range(0, my_range):

					temp_tokens = tokens.copy()
					
					if self.sentence_level:
						temp_tokens[token] = ''
						temp_instance = ' '.join(temp_tokens)
					else:
						temp_tokens[token+1] = '[UNK]' # add '' to test the classic removal process
						temp_instance = self.fix_instance(' '.join(temp_tokens[1:-1]))

					sign = 	self._find_sign(interpretation[label][token])
					#print('Token:',tokens[token+1],'Sign:',sign,'Weight:',interpretation[label][token])

					if temp_instance in self.saved_state:
						temp_prediction = self.saved_state[temp_instance]
					else:
						temp_prediction, _, _ = self.predict(temp_instance)
						self.saved_state[temp_instance] = temp_prediction
					preds = self._apply_activation(prediction, temp_prediction)
					if sign == 'positive':
						if preds[0][label] - preds[1][label] > 0:
							truthful += 1
					elif sign == 'negative':
						if  preds[1][label] -  preds[0][label] > 0:
							truthful +=1
					else:
						if preds[1][label] ==  preds[0][label]:
							truthful +=1
					#print('Prevs:',preds[0][label],'Latter:',preds[1][label])
				avg_diff.append(truthful/my_range)
			else:
				avg_diff.append(np.average([]))
		return avg_diff

	def faithful_truthfulness(self, interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales):
		avg_diff = []
		predicted_labels = self._apply_activation(prediction, prediction)[0]

		for label in range(len(self.label_names)):
			if predicted_labels[label]>=0.5: # :| h gia ola?????? pfff

				score = 0
				my_range = len(tokens) if self.sentence_level else len(tokens)-2

				for token in range(0, my_range):

					temp_tokens = tokens.copy()
					
					if self.sentence_level:
						temp_tokens[token] = ''
						temp_instance = ' '.join(temp_tokens)
					else:
						temp_tokens[token+1] = '[UNK]' # add '' to test the classic removal process
						temp_instance = self.fix_instance(' '.join(temp_tokens[1:-1]))

					sign = 	self._find_sign(interpretation[label][token])

					if temp_instance in self.saved_state:
						temp_prediction = self.saved_state[temp_instance]
					else:
						temp_prediction, _, _ = self.predict(temp_instance)
						self.saved_state[temp_instance] = temp_prediction
					preds = self._apply_activation(prediction, temp_prediction)

					if sign == 'positive':
						score += preds[0][label] - preds[1][label] 
					elif sign == 'negative':
						score += preds[1][label] - preds[0][label]
					else:
						score += (-1)*abs(preds[1][label] - preds[0][label]) #Penalty
				avg_diff.append(score)
			else:
				avg_diff.append(np.average([]))
		return avg_diff

	def faithful_truthfulness_penalty(self, interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales):
		avg_diff = []
		predicted_labels = self._apply_activation(prediction, prediction)[0]

		for label in range(len(self.label_names)):
			if predicted_labels[label]>=0.5: 

				a = interpretation[label]
				if np.unique(a).shape[0] == 1:
						value_order = {np.unique(a)[0] : a.shape[0]}
				else:
						order = np.argsort(abs(a))
						value_order = dict()
						count = 0
						size = len(a)
						for o in order:
							if abs(a[o]) in value_order:
								if value_order[abs(a[o])] < size - count:
									value_order[abs(a[o])] = size - count
							else:
								value_order[abs(a[o])] = size - count
							count = count + 1

				score = 0
				my_range = len(tokens) if self.sentence_level else len(tokens)-2

				for token in range(0, my_range):

					temp_tokens = tokens.copy()
					
					if self.sentence_level:
						temp_tokens[token] = ''
						temp_instance = ' '.join(temp_tokens)
					else:
						temp_tokens[token+1] = '[UNK]' # add '' to test the classic removal process
						temp_instance = self.fix_instance(' '.join(temp_tokens[1:-1]))

					sign = 	self._find_sign(interpretation[label][token])

					if temp_instance in self.saved_state:
						temp_prediction = self.saved_state[temp_instance]
					else:
						temp_prediction, _, _ = self.predict(temp_instance)
						self.saved_state[temp_instance] = temp_prediction
					preds = self._apply_activation(prediction, temp_prediction)
					if sign == 'positive':
						score += (preds[0][label] - preds[1][label])/value_order[abs(interpretation[label][token])]
					elif sign == 'negative':
						score += (preds[1][label] - preds[0][label])/value_order[abs(interpretation[label][token])]
					else:
						score += (-1)*abs(preds[1][label] - preds[0][label])/value_order[abs(interpretation[label][token])] #Penalty
				avg_diff.append(score)
			else:
				avg_diff.append(np.average([]))
		return avg_diff

	def robustness(self, interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales):
		avg_diff = []
		h = np.linalg.norm(hidden_states[0].mean(axis=0)-t_hidden_states[0].mean(axis=0))
		predicted_labels = self._apply_activation(prediction, prediction)[0]

		for label in range(len(self.label_names)):
			if predicted_labels[label]>=0.5:
				diff = interpretation[label]-tweaked_interpretation[label]
				norm_l2 = np.linalg.norm(np.array(diff))
				if h != 0:
					avg_diff.append(norm_l2/h)
				else:
					avg_diff.append(0)
			else:
				avg_diff.append(np.average([]))
		return avg_diff

	def auprc(self, interpretation, tweaked_interpretation, instance, prediction, tokens, hidden_states, t_hidden_states, rationales):
		aucs = []

		predicted_labels = self._apply_activation(prediction, prediction)[0]

		for label in range(len(self.label_names)):
			if predicted_labels[label]>=0.5:
				label_auc = []
				if rationales[label] != 0:
					#print(interpretation[label], rationales[label])
					precision, recall, _ = precision_recall_curve(rationales[label],interpretation[label])
					label_auc.append(auc(recall, precision))
				aucs.append(np.average(label_auc))
			else:
				aucs.append(np.average([]))
		return aucs