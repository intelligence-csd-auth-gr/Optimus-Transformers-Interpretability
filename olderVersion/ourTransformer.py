from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification


class DistilBertForMultilabelSequenceClassification(DistilBertForSequenceClassification):
	def __init__(self, config):
		super().__init__(config)

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		):
		"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		distilbert_output = self.distilbert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
		pooled_output = hidden_state[:, 0]  # (bs, dim)
		pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
		pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
		pooled_output = self.dropout(pooled_output)  # (bs, dim)
		logits = self.classifier(pooled_output)  # (bs, num_labels)

		loss = None
		if labels is not None:
			loss_fct = BCEWithLogitsLoss()
			loss = loss_fct(logits.view(-1, self.num_labels), 
							labels.float().view(-1, self.num_labels))

		if not return_dict:
			output = (logits,) + distilbert_output[1:]
			return ((loss,) + output) if loss is not None else output

		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=distilbert_output.hidden_states,
			attentions=distilbert_output.attentions,
		)

class BertForMultilabelSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
      super().__init__(config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), 
                            labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)