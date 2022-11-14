import math

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from matplotlib import cm, transforms
from torch import tensor
from torch.utils.data import Dataset as TDataset

from myEvaluation import MyEvaluation
from myExplainers import MyExplainer


def plot_sentence_heatmap(sentences, scores, title="", width=10, height=0.2, verbose=0, max_characters=143):
    """ This function was originally published in the Innvestigate library: https://innvestigate.readthedocs.io/en/latest/ but modified to work for sentences.
    Args:
		sentences: The sentences
		scores: The scores per sentence
		title: The title of the plot
		width: The width of the plot
		height: The height of the plot
		verbose: If information about the plots is going to be presented
		max_characters: The max characters per row
    """
    fig = plt.figure(figsize=(width, height), dpi=200)

    ax = plt.gca()
    ax.set_title(title, loc='left')

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)

    canvas = ax.figure.canvas
    t = ax.transData

    # normalize scores to the followings:
    # - negative scores in [0, 0.5]
    # - positive scores in (0.5, 1]
    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    characters_per_line = 0
    loc_y = -0.2
    for i, token in enumerate(sentences):
        if token == '.':
            score = 0.5
        else:
            score = normalized_scores[i]
        *rgb, _ = cmap.to_rgba(score, bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)

        bbox_options = {'facecolor': color, 'pad': -0.1, 'linewidth': 0,
                        'boxstyle': 'round,pad=0.44,rounding_size=0.2'}
        text = ax.text(0.0, loc_y, token, bbox=bbox_options, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        if characters_per_line + len(token) + 1 >= max_characters + 1:
            loc_y = loc_y - 1.8
            t = ax.transData
            characters_per_line = len(token) + 1
        else:
            t = transforms.offset_copy(
                text._transform, x=ex.width+15, units='dots')
            characters_per_line = characters_per_line + len(token) + 1

    ax.axis('off')


def plot_text_heatmap(words, scores, title="", width=10, height=0.2, verbose=0, max_word_per_line=20):
    """ This function was originally published in the Innvestigate library: https://innvestigate.readthedocs.io/en/latest/ but modified to work for sentences.
    Args:
		words: The tokens of the input
		scores: The scores per sentence
		title: The title of the plot
		width: The width of the plot
		height: The height of the plot
		verbose: If information about the plots is going to be presented
		max_word_per_line: The max words per row
    """
    fig = plt.figure(figsize=(width, height), dpi=150)

    ax = plt.gca()
    ax.set_title(title, loc='left')
    tokens = words

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)

    canvas = ax.figure.canvas
    t = ax.transData

    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5

    loc_y = -0.2
    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)

        bbox_options = {'facecolor': color, 'pad': 3.7,
                        'linewidth': 0, 'boxstyle': 'round,pad=0.37'}
        text = ax.text(0.0, loc_y, token, bbox=bbox_options, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()

        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y - 2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(
                text._transform, x=ex.width+15, units='dots')

    ax.axis('off')


class MyDataset(TDataset):
    """MyDataset class is used to transform an instance (input sequence) to be appropriate for use in transformers
    """

    def __init__(self, encodings, labels, tokenizer):
        self.encodings = tokenizer(
            list(encodings), truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Optimus:
    """Optimus class is an easy-to-use tool for interpreting locally transformer models (currently BERT and DistilBERT). Check the github's repo for a few examples!"""

    def __init__(self, model, tokenizer, label_names, task, set_of_instance=None):
        """ Init function
        Args:
			model: The loaded model. You should use the MyModel class to load it (Please check and use the MyModel script)
			tokenizer: The tokenizer
			label_names: The label names
			task: 'single_label' or 'multi_label' task
			set_of_instance: A set of instances for Optimus Batch
        Attributes:
			model: The huggingface trainer exported by the given MyModel
			other_model: The actual MyModel

			key_list: The list of the key-related linear layers -> initiated through the __get_additional_info_from_trainer__ function
			query_list: The list of the query-related linear layers -> initiated through the __get_additional_info_from_trainer__ function
			layers: The number of layers
			heads: The number of heads
			embedding_size: The size of the embedding
			ehe: The size of the embedding per head
        """
        self.model = model.trainer
        self.other_model = model
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.task = task
        self.set_of_instance = set_of_instance
        self.__export_model_information__()
        if len(self.set_of_instance) > 1:
            self.__identify_max_across__()
            print('Best setup in batch set calculated!')
        else:
            print('No best setup calculated as you did not provided a set of instances. You will not be able to use max_across later')

    def __export_model_information__(self):
        """This function initialize parameters such as number of heads, layers, embedding size, according to the model.
        key_list and query_list contain the layers of the models that are needed to recreate the attention matrices using the hidden states
        """
        self.key_list = []
        self.query_list = []
        if 'distilbert' in str(type(self.model.model)).lower():
            self.model_name = 'distilbert'
        else:
            self.model_name = 'bert'

        if self.model_name.lower() == 'bert':
            self.layers = 12
            self.heads = 12
            self.embedding_size = 768
            self.ehe = 64  # 768/12
            for i in range(self.layers):
                self.key_list.append(
                    self.model.model.base_model.encoder.layer[i].attention.self.key.weight.cpu().detach().numpy())
                self.query_list.append(
                    self.model.model.base_model.encoder.layer[i].attention.self.query.weight.cpu().detach().numpy())
        elif self.model_name.lower() == 'distilbert':
            self.layers = 6
            self.heads = 12
            self.embedding_size = 768
            self.ehe = 64  # 768/12
            for i in range(self.layers):
                self.key_list.append(
                    self.model.model.distilbert.transformer.layer[i].attention.k_lin.weight.cpu().detach().numpy())
                self.query_list.append(
                    self.model.model.distilbert.transformer.layer[i].attention.q_lin.weight.cpu().detach().numpy())
        else:
            print('Currently, this implementation works for BERT/DistilBERT only. You provided another model, therefore explain() function will not work.')

    def __identify_configurations__(self, raw_attention='A'):
        """ This function calculates the all the available combination for the attention matrices operations (head, layer, matrix level)
		Args:
			raw_attention: Describes if the combinations will be made for A or A*. In A* multi is not used in heads
		Return:
			conf: The identified configurations
        """
        if raw_attention == 'A':
            conf = []
            for ci in ['Mean', 'Multi'] + list(range(self.layers)):
                for ce in ['Mean'] + list(range(self.heads)):
                    # Matrix: From, To, MeanColumns, MeanRows, MaxColumns, MaxRows
                    for cp in ['From', 'To', 'MeanColumns', 'MaxColumns']:
                        for cl in [False]:  # Selection: True: select layers per head, False: do not
                            conf.append([ci, ce, cp, cl])
        else:
            conf = []
            for ci in ['Mean'] + list(range(self.layers)):
                for ce in ['Mean'] + list(range(self.heads)):
                    # Matrix: From, To, MeanColumns, MeanRows, MaxColumns, MaxRows
                    for cp in ['From', 'To', 'MeanColumns', 'MaxColumns']:
                        for cl in [False]:  # Selection: True: select layers per head, False: do not
                            conf.append([ci, ce, cp, cl])
        return conf

    def __inference__(self, instance, raw_attention='A'):
        """ This function given an instance makes a prediction based on the model, as well as it produces the attention matrices
		Args:
				instance: The examined instance ('string')
				raw_attention: Describes if the combinations will be made for A or A*. In A* multi is not used in heads
		Return:
				predictions[0]: The predictions regarding the examined instance
				attention: The attention matrices
        """
        instance = [instance, instance]
        if self.task == 'single_label':
            instance_labels = [0, 0]
        else:
            instance_labels = [
                [0] * len(self.label_names), [0] * len(self.label_names)]
        instance_dataset = MyDataset(instance, instance_labels, self.tokenizer)
        outputs = self.model.predict(instance_dataset)
        predictions = outputs.predictions[0]
        hidden_states = np.array(list(outputs.predictions[1]))[:, 0, :, :]

        attention = []
        for la in range(self.layers):
            our_new_layer = []
            if self.model_name.lower() == 'bert':
                bob = self.model.model.base_model.encoder.layer[la].attention
                has = hidden_states[la]
                aaa = bob.self.key(torch.tensor(has).to('cuda'))
                bbb = bob.self.query(torch.tensor(has).to('cuda'))
            else:
                bob = self.model.model.base_model.transformer.layer[la].attention
                has = hidden_states[la]
                aaa = bob.k_lin(torch.tensor(has).to('cuda'))
                bbb = bob.q_lin(torch.tensor(has).to('cuda'))
            for he in range(self.heads):
                if self.model_name.lower() == 'bert':
                    attention_scores = torch.matmul(
                        bbb[:, he*64:(he+1)*64], aaa[:, he*64:(he+1)*64].transpose(-1, -2))
                    attention_scores = attention_scores / math.sqrt(64)
                else:
                    bbb = bbb / math.sqrt(64)
                    attention_scores = torch.matmul(
                        bbb[:, he*64:(he+1)*64], aaa[:, he*64:(he+1)*64].transpose(-1, -2))
                if raw_attention == 'A':
                    attention_scores = torch.nn.functional.softmax(
                        attention_scores, dim=-1)
                our_new_layer.append(attention_scores.cpu().detach().numpy())
            attention.append(our_new_layer)
        attention = np.array(attention)
        return predictions[0], attention

    def __pseudo_predict__(self, instance):
        """ This function is a pseudo predict function, which returns the output of inference in an acceptable format.
		Args:
				instance: The examined instance ('string')
		Return:
				predictions[0]: The predictions regarding the examined instance
				attention: The attention matrices
        """
        result = self.__inference__(instance)
        return result[0], result[1], 'None'

    def explain(self, instance, mode='baseline', level='token', raw_attention='A'):
        """ This function given an input and a few parameters produces the explanation using Optimus
		Args:
				instance: The examined instance ('string')
				mode: can be either 'baseline' (baseline setup mean,mean,from), 'max_across' (Optimus Batch), 'max_per_instance' (Optimus Prime), 'max_per_instance_per_label' (Optimus Label)
				level: 'token' (for token-level interpretations) or 'sentence' (sentence-level interpretations)
				raw_attention: Describes if the combinations will be made for A or A*. In A* multi is not used in heads. With A* the interpretations can include negative values
		Return:
				interpretations[0]: The interpretations regarding the examined instance for all labels
				tokens: The instance tokenized either on token-level or on sentence-level
        """
        if self.set_of_instance == None:
            if mode not in ['baseline', 'max_per_instance', 'max_per_instance_per_label']:
                print(
                    'Please choose a mode among: baseline, max_per_instance, max_per_instance_per_label')
                return
        else:
            if mode not in ['baseline', 'max_across', 'max_per_instance', 'max_per_instance_per_label']:
                print(
                    'Please choose a mode among: baseline, max_across, max_per_instance, max_per_instance_per_label')
                return
        if level not in ['token', 'sentence']:
            print('Please choose a level among: token, sentence')
            return
        if raw_attention not in ['A', 'A*']:
            print('Please choose raw attention among: A (for attention after softmax), A* (for attention before softmax)')
            return

        if level == 'token':
            my_evaluators = MyEvaluation(
                self.label_names, self.__pseudo_predict__, False, True)
            my_explainers = MyExplainer(self.label_names, self.other_model)
        else:
            my_evaluators = MyEvaluation(
                self.label_names, self.__pseudo_predict__, True, True)
            my_explainers = MyExplainer(
                self.label_names, self.other_model, True, '‡')

        if raw_attention == 'A':
            conf = self.__identify_configurations__()
        else:
            conf = self.__identify_configurations__('A*')

        # This is RFT from the paper
        evaluation = {'FTP': my_evaluators.faithful_truthfulness_penalty}
        metrics = {'FTP': []}

        prediction, attention = self.__inference__(instance, raw_attention)
        enc = self.tokenizer([instance, instance],
                             truncation=True, padding=True)[0]
        mask = enc.attention_mask
        tokens = enc.tokens
        temp_tokens = enc.tokens
        interpretations = []
        for con in conf:
            my_explainers.config = con
            temp = my_explainers.my_attention(
                instance, prediction, temp_tokens, mask, attention, None)
            if level == 'sentence':
                tokens = temp[0].copy()[0]
                temp = temp[1].copy()
            if raw_attention == 'A':
                temp = [sklearn.preprocessing.minmax_scale(
                    i, feature_range=(0.0000001, 1)) for i in temp]
            if raw_attention == 'A*':
                for t in range(len(temp)):
                    temp[t] = np.array(temp[t])/np.max(abs(np.array(temp[t])))
            interpretations.append(temp)

        if mode == 'baseline':
            return interpretations[0], tokens
        elif mode == 'max_across':
            if raw_attention == 'A':
                return interpretations[self.max_across_a[level]], tokens
            else:
                return interpretations[self.max_across_a_star[level]], tokens
        elif mode in ['max_per_instance', 'max_per_instance_per_label']:
            for metric in metrics.keys():
                evaluated = []
                for interpretation in interpretations:
                    evaluated.append(evaluation[metric](
                        interpretation, None, instance, prediction, tokens, None, None, None))
                metrics[metric].append(evaluated)
            rft = np.array(metrics['FTP'])
            if mode == 'max_per_instance':
                rft = rft.mean(axis=1)
                max_id = np.argmax(rft)
                return interpretations[max_id], tokens
            elif mode == 'max_per_instance_per_label':
                temp_interpretations = []
                for label in range(len(self.label_names)):
                    id_max = np.argmax(rft[0, :, label])
                    temp_interpretations.append(interpretations[id_max][0])
                return temp_interpretations, tokens
        return interpretations[0], tokens

    def __identify_max_across__(self):
        """ This function given an evaluation set identifies the best combination of attention operations.
		Attributes:
				max_across_a: A dictionary containing the id of the best combination for both token and sentence-level with the classic attention matrices
				max_across_a_star: A dictionary containing the id of the best combination for both token and sentence-level with the modified attention matrices (A*)
        """
        conf_a = self.__identify_configurations__()
        conf_a_star = self.__identify_configurations__('A*')

        self.max_across_a = {'token': 0, 'sentence': 0}
        self.max_across_a_star = {'token': 0, 'sentence': 0}

        for level in ['token', 'sentence']:
            if level == 'token':
                my_evaluators = MyEvaluation(
                    self.label_names, self.__pseudo_predict__, False, True)
                my_explainers = MyExplainer(self.label_names, self.other_model)
            else:
                my_evaluators = MyEvaluation(
                    self.label_names, self.__pseudo_predict__, True, True)
                my_explainers = MyExplainer(
                    self.label_names, self.other_model, True, '‡')

            evaluation = {'FTP': my_evaluators.faithful_truthfulness_penalty}
            metrics = {'FTP': []}

            for instance in self.set_of_instance:
                my_evaluators.clear_states()
                my_explainers.save_states = {}
                prediction, attention = self.__inference__(instance, 'A')
                enc = self.tokenizer([instance, instance],
                                     truncation=True, padding=True)[0]
                mask = enc.attention_mask
                tokens = enc.tokens
                temp_tokens = tokens.copy()
                interpretations = []
                if (tokens.count('.') >= 2 and level == 'sentence') or level == 'token':
                    for con in conf_a:
                        my_explainers.config = con
                        temp = my_explainers.my_attention(
                            instance, prediction, tokens, mask, attention, None)
                        if level == 'sentence':
                            temp_tokens = temp[0].copy()[0]
                            temp = temp[1].copy()
                        interpretations.append([sklearn.preprocessing.minmax_scale(
                            i, feature_range=(0.0000001, 1)) for i in temp])
                    for metric in metrics.keys():
                        evaluated = []
                        for interpretation in interpretations:
                            evaluated.append(evaluation[metric](
                                interpretation, None, instance, prediction, temp_tokens, None, None, None))
                        metrics[metric].append(evaluated)

            confs_mean = []
            temp_metric = np.array(metrics['FTP'])
            if len(temp_metric) == 0:
                self.max_across_a[level] = 'Not Calculated'
            else:
                for i in range(len(conf_a)):
                    label_score = []
                    for label in range(len(self.label_names)):
                        tempo = [k for k in temp_metric[:, i, label]
                                 if str(k) != str(np.average([]))]
                        if len(tempo) == 0:
                            tempo.append(0)
                        label_score.append(np.array(tempo))
                    temp_mean = []
                    for k in label_score:
                        temp_mean.append(k.mean())
                    confs_mean.append(np.array(temp_mean).mean())
                self.max_across_a[level] = np.argmax(confs_mean)

            metrics = {'FTP': []}

            for instance in self.set_of_instance:
                my_evaluators.clear_states()
                my_explainers.save_states = {}
                prediction, attention = self.__inference__(instance, 'A*')
                enc = self.tokenizer([instance, instance],
                                     truncation=True, padding=True)[0]
                mask = enc.attention_mask
                tokens = enc.tokens
                temp_tokens = tokens.copy()
                interpretations = []
                if (tokens.count('.') >= 2 and level == 'sentence') or level == 'token':
                    for con in conf_a_star:
                        my_explainers.config = con
                        temp = my_explainers.my_attention(
                            instance, prediction, tokens, mask, attention, None)
                        if level == 'sentence':
                            temp_tokens = temp[0].copy()[0]
                            temp = temp[1].copy()
                        interpretations.append(
                            [np.array(i)/np.max(abs(np.array(i))) for i in temp])
                    for metric in metrics.keys():
                        evaluated = []
                        for interpretation in interpretations:
                            evaluated.append(evaluation[metric](
                                interpretation, None, instance, prediction, temp_tokens, None, None, None))
                        metrics[metric].append(evaluated)

            confs_mean = []
            temp_metric = np.array(metrics['FTP'])
            if len(temp_metric) == 0:
                self.max_across_a_star[level] = 'Not Calculated'
            else:
                for i in range(len(conf_a_star)):
                    label_score = []
                    for label in range(len(self.label_names)):
                        tempo = [k for k in temp_metric[:, i, label]
                                 if str(k) != str(np.average([]))]
                        if len(tempo) == 0:
                            tempo.append(0)
                        label_score.append(np.array(tempo))
                    temp_mean = []
                    for k in label_score:
                        temp_mean.append(k.mean())
                    confs_mean.append(np.array(temp_mean).mean())
                self.max_across_a_star[level] = np.argmax(confs_mean)
