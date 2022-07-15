# Is Attention Enough for Interpreting Transformers in Text Classification?
Transformers are widely used in NLP, where they consistently achieve state-of-the-art performance. This is due to their attention-based architecture, which allows them to model rich linguistic relations between words. However, transformers are difficult to interpret. Being able to provide reasoning for its decisions is an important property for a model in domains where human lives are affected, such as hate speech detection and biomedicine. With transformers finding wide use in these fields, the need for interpretability techniques tailored to them arises. The effectiveness of attention-based interpretability techniques for transformers in text classification is studied in this work. Despite concerns about attention-based interpretations in the literature, we show that, with proper setup, attention may be used in such tasks with results comparable to state-of-the-art techniques, while also being faster and friendlier to the environment. We validate our claims with a series of experiments that employ a new feature importance metric.

Regarding the trained ML models, a note with the repo containing them will be added in the folder models after publication.

## Instructions
Please ensure you have docker installed on your desktop. If you want to build the docker by yourself please download this repo.
 the folder lionets with the requirements.txt file and name the zip "lionets.zip". Then:
```bash
cd repo/docker 
docker build -t vizualization_tool .
```
After succesfully building the image, please do:
```bash
docker run -p 8866:8866 vizualization_tool
```
By accessing your localhost in the port 8866 (http://localhost:8866), you will be able to check and use the docker. 

The ready dockerized version will be available in docker hub after publication.

## Contributors on Ethos
Name | Email
--- | ---
Anonymous | Anonymous

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
