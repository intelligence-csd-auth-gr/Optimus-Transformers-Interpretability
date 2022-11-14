from transformers import Trainer, TrainingArguments, utils
import numpy as np
from torch.utils.data import Dataset as TDataset
from torch import tensor
from tensorflow.keras.layers import Softmax
from tensorflow.keras.activations import tanh

class MyDataset(TDataset):
	def __init__(self, encodings, labels, tokenizer):
		self.encodings = tokenizer(list(encodings), truncation=True, padding=True)
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)

class MyModel:
	def __init__(self, path, dataset_name, model_name, task, labels, cased):
		self.path = path
		self.dataset_name = dataset_name
		self.model_name = model_name
		self.task = task
		self.labels = labels
		self.cased = cased
		self.__load_model__()
		self.__get_additional_info_from_trainer__()
	
	def __load_model__(self):
		if self.model_name.lower() == 'bert':
			from transformers import BertTokenizerFast
			if self.task.lower() == 'single_label':
				from transformers import BertForSequenceClassification as transformer_model
			else:
				from ourTransformer import BertForMultilabelSequenceClassification as transformer_model
			if self.cased:
				self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
			else:
				self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
		elif self.model_name.lower() == 'distilbert':
			from transformers import DistilBertTokenizerFast
			if self.task.lower() == 'single_label':
				from transformers import DistilBertForSequenceClassification as transformer_model
			else:
				from ourTransformer import DistilBertForMultilabelSequenceClassification as transformer_model
			if self.cased:
				self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
			else:
				self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
		if self.task.lower() == 'single_label':
			model = transformer_model.from_pretrained(self.path+self.dataset_name, output_attentions=True,
															output_hidden_states=True)
		else:
			model = transformer_model.from_pretrained(self.path+self.dataset_name, num_labels = self.labels ,output_attentions=True,
															output_hidden_states=True)
		#utils.logging.disable_progress_bar() 
		training_arguments = TrainingArguments(
			evaluation_strategy='epoch',     # evaluation frequency
			save_strategy='epoch',           # model checkpoint frequency
			logging_strategy='epoch',        # logging frequency
			log_level='critical',             # logging level
			output_dir='./results',          # output directory
			num_train_epochs=1,              # total number of training epochs
			per_device_train_batch_size=4,  # batch size per device during training
			per_device_eval_batch_size=4,   # batch size for evaluation
			warmup_steps=200,                # number of warmup steps for learning rate scheduler
			weight_decay=0.01,               # strength of weight decay
			logging_dir='./logs'             # directory for storing logs
		)
		self.trainer = Trainer(model=model, args=training_arguments)

	def __get_additional_info_from_trainer__(self):
		self.label_weights = np.array(self.trainer.model.classifier.weight.tolist())
		self.key_list = []
		self.query_list = []
		if self.model_name.lower() == 'bert':
			self.layers = 12
			self.embedding_size = 768
			for i in range(self.layers):
				self.key_list.append(self.trainer.model.base_model.encoder.layer[i].attention.self.key.weight.cpu().detach().numpy())
				self.query_list.append(self.trainer.model.base_model.encoder.layer[i].attention.self.query.weight.cpu().detach().numpy())
		else:
			self.layers = 6
			self.embedding_size = 512
			for i in range(self.layers):
				self.key_list.append(self.trainer.model.distilbert.transformer.layer[i].attention.k_lin.weight.cpu().detach().numpy())
				self.query_list.append(self.trainer.model.distilbert.transformer.layer[i].attention.q_lin.weight.cpu().detach().numpy())
		 
	def my_predict(self, instance):
		instance = [instance,instance]
		if self.task == 'single_label':
			instance_labels = [0, 0]
		else:
			instance_labels = [[0]* self.labels,[0]*self.labels]
		instance_dataset = MyDataset(instance, instance_labels, self.tokenizer)
		outputs = self.trainer.predict(instance_dataset)
		predictions = outputs.predictions[0]
		hidden_states = np.array(list(outputs.predictions[1]))
		attention_matrix = np.array(list(outputs.predictions[2]))
		#get_additional_info_from_trainer(model_name)
		return predictions[0], attention_matrix[:,0,:,:], hidden_states[:,0,:,:]