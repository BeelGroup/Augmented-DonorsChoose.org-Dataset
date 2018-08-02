# Code and Literature Repository for Investigating Meta-Learning Algorithm in the Context of Recommender Systems

## Inspiration

The main inspiration for this research is based on the work performed by the ADAPT group at the Trinity college in Dublin. Of special note for this project is the research outlined in the paper [One-at-a-time: A Meta-Learning Recommender-System for Recommendation-Algorithm Selection on Micro Level](https://arxiv.org/abs/1805.12118).

## Roadmap

* Find a suitable dataset for meta-learning
  * Candidates should provide information about the user, the item and about the context of each transaction
  * Adequate sources might be [kaggle](https://www.kaggle.com), [Google public datasets](https://cloud.google.com/public-datasets/) and previous [RecSys challenges](https://recsys.acm.org/)
* Evaluate existing software frameworks for their applicability as meta-feature generators
  * Meta-feature algorithms should include collaborative, content based and possibly deep learning based approaches
  * Suitable frameworks might be Tensorflow, scikit-learn and higher level libraries like Keras and scikit-surprise
* Train and compare various meta-learning models
  * Predict either rating error or reformulate algorithm selection as classification problem
  * Evaluate model using appropriate variables, possible candidates might be the normalized discounted cumulative gain or the Kendall rank correlation coefficient