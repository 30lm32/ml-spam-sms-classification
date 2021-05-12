![Image](https://appliedmachinelearning.files.wordpress.com/2017/01/spam-filter.png)


### Which one does it catch whole* SPAM SMS?

|__Problem__|__Data__|__Methods__|__Libs__|__Link__|
|-|-|-|-|-|
|`NLP`|Text|`Naive Bayesian`, `SVM`, `Random Forest Classifier`, `Deep Learning - LSTM`, `Word2Vec`|`Sklearn`, `Keras`, `Gensim`, `Pandas`, `Seaborn`|https://github.com/erdiolmezogullari/ml-spam-sms-classification|

If you want to see the further ML projects, you may visit my main repo: https://github.com/erdiolmezogullari/ml-projects

In this project, We applied supervised learning (classification) algorithms and deep learning (LSTM).

We used a public [SMS Spam dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection), which is not a purely clean dataset. The data consists of two different columns (features), such as context, and class. The column context is referring to SMS. The column class may take a value that can be either `spam` or `ham` corresponding to related SMS context.

Before applying any supervised learning methods, we applied a bunch of data cleansing operations to get rid of messy and dirty data since it has some broken and messy context.

After obtaining the cleaned dataset, we created tokens and lemmas of SMS corpus separately by using [Spacy](https://spacy.io/), and then, we generated [bag-of-word](https://en.wikipedia.org/wiki/Bag-of-words_model) and [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) of SMS corpus, respectively. In addition to these data transformations, we also performed [SVD](https://en.wikipedia.org/wiki/Singular-value_decomposition), [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to reduce dimension of dataset.

To manage data transformation in the training and testing phase effectively and avoid [data leakage](https://www.kaggle.com/wiki/Leakage), we used Sklearn's [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) class. So, we added each data transformation step (e.g. `bag-of-word`, `TF-IDF`, `SVC`) and classifier (e.g. `Naive Bayesian`, `SVM`, `Random Forest Classifier`) into an instance of class `Pipeline`.

After applying those supervised learning methods, we also performed deep learning.
The deep learning architecture we used is based on [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory). To perform LSTM approaching in [Keras  (Tensorflow)](https://keras.io/), we needed to create an embedding matrix of our corpus. So, we used [Gensim's Word2Vec](https://radimrehurek.com/gensim/) approach to obtain embedding matrix, rather than TF-IDF.

At the end of each processing by using a different classifier, we plotted [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) to compare which one the best classifier for filtering SPAM SMS.
