# Optimus

This GitHub contains the code of the experiments for our paper, and an example on how to easily use it in your pipeline. The folder "[olderVersion](https://github.com/intelligence-csd-auth-gr/Optimus-Transformers-Interpretability/tree/main/olderVersion)" contains the code for the [preliminary version of our paper](https://arxiv.org/abs/2209.10876). 

<table align="center">
    <tr>
        <td> <img src="https://github.com/intelligence-csd-auth-gr/Optimus-Transformers-Interpretability/blob/main/logo.png?raw=true" width="250"  height="250"></td>
        <td align="center"><p><h2>Optimus Tranformers Interpretability</h2><h4>Local Attention-based Interpretation of Transformers in Text Classification</h4></p></td>
    </tr>
</table>

## Abstract
Transformers are widely used in natural language processing, where they consistently achieve state-of-the-art performance. This is mainly due to their attention-based architecture, which allows them to model rich linguistic relations between (sub)words. However, transformers are difficult to interpret. Being able to provide reasoning for its decisions is an important property for a model in domains where human lives are affected. With transformers finding wide use in such fields, the need for interpretability techniques tailored to them arises. We propose a new technique that selects the most faithful attention-based interpretation among the several ones that can be obtained by combining different head, layer and matrix operations. In addition, two variations are introduced towards (i) reducing the computational complexity, thus being faster and friendlier to the environment, and (ii) enhancing the performance in multi-label data. We further propose a new faithfulness metric that is more suitable for transformer models and exhibits high correlation with the area under the precision-recall curve based on ground truth rationales. We validate our claims with a series of quantitative and qualitative experiments on seven datasets.

## Requirements

For the requirements just check the req.txt file.

## Example
```python
#Load your transformer model (e.g. DistilBERT) using MyModel class
model_path = 'Trained Models/' 
model = 'distilbert_hx' 
model_name = 'distilbert' 
task = 'single_label' 
labels = 2 
cased = False 
model = MyModel(model_path, model, model_name, task, labels, cased)
tokenizer = model.tokenizer #Extract your tokenizer

#Load your dataset
hx = Dataset(path=data_path)
x, y, label_names, rationales = hx.load_hatexplain(tokenizer)

#Load Optimus class
from optimus import Optimus, plot_text_heatmap
ionbot = Optimus(model, tokenizer, label_names, task, set_of_instance=[])
#Leave set_of_instances empty (empty list) because it takes a large amount of time to calculate the best configuration for the given set (Optimus Batch). Use it only if you want to later use Optimus Batch, to lower computational complexity during runtime.

#Then select a random instance
instance = "This sentence contains hate speech content for ****** people!"
prediction, attention, hidden_states = model.my_predict(instance) #use MyModel instance to make a prediction

#Finally, use Optimus to extract interpretations: a) first using the simple baseline attention setup (mean, mean, from), b) then using Optimus Prime and c) Optimus Label.
baseline = ionbot.explain(instance, mode='baseline', level='token', raw_attention='A')
per_instance = ionbot.explain(instance, mode='max_per_instance', level='token', raw_attention='A')
per_instance_per_label = ionbot.explain(instance, mode='max_per_instance_per_label', level='token', raw_attention='A')

#You can also present them using the plot_text_heatmap funtion:
selected_label = 1
tokens = baseline[1] #The output of Optimus contains (interpretations, tokens/sentences)
plot_text_heatmap(tokens[1:-1], np.array(baseline[0][selected_label]))
plot_text_heatmap(tokens[1:-1], np.array(per_instance[0][selected_label]))
plot_text_heatmap(tokens[1:-1], np.array(per_instance_per_label[0][selected_label]))

#Tips: a) if you want to use raw attention without softmax, use A* instead of A in raw_attention, b) if you want sentence level use sentence in level variable, and use the plot_sentence_heatmap function.
```

## Developed by: 
|    Name             |      e-mail          |
| --------------------| -------------------- |
| Nikolaos Mylonas    | myloniko@csd.auth.gr |
| Ioannis Mollas      | iamollas@csd.auth.gr |
| Grigorios Tsoumakas |  greg@csd.auth.gr    |


## Funded by
The research work was supported by the Hellenic Foundation forResearch and Innovation (H.F.R.I.) under the “First Call for H.F.R.I.Research Projects to support Faculty members and Researchers and the procurement of high-cost research equipment grant” (ProjectNumber: 514).


## Additional resources
- [AMULET project](https://www.linkedin.com/showcase/amulet-project/about/)
- [Academic Team's page](https://intelligence.csd.auth.gr/#)
 
 ![amulet-logo](https://user-images.githubusercontent.com/6009931/87019683-9204ad00-c1db-11ea-9394-855d1d3b41b3.png)

 ## Citation
Please cite the paper if you use it in your work or experiments :D :

- [Journal] :
    - TBA
- [Preprint] 
    - https://arxiv.org/abs/2209.10876, available on arxiv
