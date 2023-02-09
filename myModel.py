import numpy as np
from torch import tensor
from torch.utils.data import Dataset as TDataset
from transformers import Trainer, TrainingArguments


class MyDataset(TDataset):
	"""MyDataset class is used to transform an instance (input sequence) to be appropriate for use in transformers
	"""
	def __init__(self, encodings, labels, tokenizer):
		self.encodings = tokenizer(
			list(encodings), truncation=True, padding=True)
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: tensor(val[idx])
				for key, val in self.encodings.items()}
		item['labels'] = tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)


class MyModel:
	"""MyModel class loads the transformer model, and setups the myPredict function to later use in the explanation techniques"""

	def __init__(self, path, dataset_name, model_name, task, labels, cased):
		"""Init function
		Args:
			path: The path of the folder with the trained models
			dataset_name: The name of the dataset to be used in conjunction to the path to find and load the model
			model_name: The transformer's name (currently only 'bert' and 'distilbert' are available)
			task: 'single_label' or 'multi_label' task
			labels: The number of the labels from our dataset (integer)
			cased: Boolean for cased (True) or uncased (False)
		Attributes:
			trainer: The huggingface trainer module (which includes the loaded model) -> initiated through the __load_model__ function
			tokenizer: The model specific tokenizer -> initiated through the __load_model__ function
			key_list: The list of the key-related linear layers -> initiated through the __get_additional_info_from_trainer__ function
			query_list: The list of the query-related linear layers -> initiated through the __get_additional_info_from_trainer__ function
			layers: The number of layers
			heads: The number of heads
			embedding_size: The size of the embedding
			ehe: The size of the embedding per head
		"""
		self.path = path
		self.dataset_name = dataset_name
		self.model_name = model_name
		self.task = task
		self.labels = labels
		self.cased = cased
		self.__load_model__()
		self.__get_additional_info_from_trainer__()

	def __load_model__(self):
		"""This function identifies and loads the correct fine-tuned model
		"""
		if self.cased:
			cs = '-cased'
		else:
			cs = '-uncased'
		if self.model_name.lower() == 'bert':
			from transformers import BertTokenizerFast
			self.tokenizer = BertTokenizerFast.from_pretrained(
				'bert-base'+cs)
			if self.task.lower() == 'single_label':
				from transformers import \
					BertForSequenceClassification as transformer_model
			else:
				from myTransformer import \
					BertForMultilabelSequenceClassification as \
					transformer_model
		elif self.model_name.lower() == 'distilbert':
			from transformers import DistilBertTokenizerFast
			self.tokenizer = DistilBertTokenizerFast.from_pretrained(
				'distilbert-base'+cs)
			if self.task.lower() == 'single_label':
				from transformers import \
					DistilBertForSequenceClassification as \
					transformer_model
			else:
				from myTransformer import \
					DistilBertForMultilabelSequenceClassification as \
					transformer_model
		if self.task.lower() == 'single_label':
			model = transformer_model.from_pretrained(self.path+self.dataset_name, output_attentions=True,
														output_hidden_states=True)
		else:
			model = transformer_model.from_pretrained(self.path+self.dataset_name, num_labels = self.labels, output_attentions=True,
														output_hidden_states=True)
		# utils.logging.disable_progress_bar() #Enable this line to allow it to run in terminal. Comment this line to run it in notebooks/colab
		training_arguments = TrainingArguments(evaluation_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
												log_level='critical', output_dir='./results', num_train_epochs=1,
												per_device_train_batch_size=8, per_device_eval_batch_size=8,
												warmup_steps=200, weight_decay=0.01, logging_dir='./logs'
												)
		self.trainer = Trainer(model=model, args=training_arguments)

	def __get_additional_info_from_trainer__(self):
		"""This function initialize parameters such as number of heads, layers, embedding size, according to the model. 
		key_list and query_list contain the layers of the models that are needed to recreate the attention matrices using the hidden states
		"""
		self.key_list = []
		self.query_list = []
		if self.model_name.lower() == 'bert':
			self.layers = 12
			self.heads = 12
			self.embedding_size = 768
			self.ehe = 64  # 768/12
			for i in range(self.layers):
				self.key_list.append(
					self.trainer.model.base_model.encoder.layer[i].attention.self.key.weight.cpu().detach().numpy())
				self.query_list.append(
					self.trainer.model.base_model.encoder.layer[i].attention.self.query.weight.cpu().detach().numpy())
		else:
			self.layers = 6
			self.heads = 12
			self.embedding_size = 768
			self.ehe = 64  # 768/12
			for i in range(self.layers):
				self.key_list.append(
					self.trainer.model.distilbert.transformer.layer[i].attention.k_lin.weight.cpu().detach().numpy())
				self.query_list.append(
					self.trainer.model.distilbert.transformer.layer[i].attention.q_lin.weight.cpu().detach().numpy())

	def my_predict(self, instance):
		"""This function allows the prediction for a single instance (a single input sequence). 
		It returns the prediction, the attention matrices, and the hidden states
		"""
		# We double the instance, because the trainer module needs a "dataset"
		instance = [instance, instance]
		# We also add dummy labels, which are not used, but needed by the module
		if self.task == 'single_label':
			instance_labels = [0, 0]
		else:
			instance_labels = [[0] * self.labels, [0]*self.labels]
		instance_dataset = MyDataset(
			instance, instance_labels, self.tokenizer)
		outputs = self.trainer.predict(instance_dataset)
		predictions = outputs.predictions[0]
		hidden_states = np.array(list(outputs.predictions[1]))
		attention_matrix = np.array(list(outputs.predictions[2]))
		return predictions[0], attention_matrix[:, 0, :,:], hidden_states[:,0,:,:]
