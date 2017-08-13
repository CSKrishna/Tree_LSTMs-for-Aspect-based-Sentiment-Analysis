# Tree_LSTMs
In this code base, we implement a Constituency Tree-LSTM model for sentence level aspect based sentiment analysis (ABSA). The training/validation dataset for the model consists of sentences from a domain (laptop reviews, hotel reviews etc). Each sentence from the training dataset is annotated with the aspect and polarity associated with it. For example, a labelled instance from the Laptop domain would be:
<sentence id="B00KMRGF28_381_AH6TXTDWVUNLS:0>
<text> Fantastic for the price, its a pity keys were not
illuminated. </text>
<Opinions>
<Opinion category="LAPTOP#PRICE" polarity="positive"/>
<Opinion category="KEYBOARD#DESIGN_FEATURES" polarity="negative"/>
</Opinions>
</sentence>
The model need to be trained over these annotated sentences so that it can, for a new sentence, a) output the list of aspects present in it b) predict the sentiment polarity (negative, positive, mildly negative/positive) associated with it.
Laptop Review Dataset & Preprocessing
For model training and validation, we use the Laptop review data set. This dataset comprises 3048 sentences extracted from customer reviews of laptops and spans 154 aspects. Out of these, the 17 most frequently occurring aspects account for ~80% of the aspect labels in the dataset. So as a pre-processing step, we collapse the remaining aspects into a miscellaneous category ‘Other’ to bound model complexity and avoid over-fitting. Therefore, in the revised task, the model has to predict the presence/absence of 18 aspects in a sentence.
We use the Stanford NLP parser to generate binary parse trees for each sentence in the training set as part of an offline pre-processing step (In subsequent versions, we’ll include the code to generate binary parse trees). The aspect-polarity annotations are appended as an 18 character string to the root node, where the position in the string signifies the aspect, and the character – ‘1’ for positive sentiment, 2 for mildly positive/negative, 3 for negative, 0  for missing aspect - encodes the polarity or lack thereof . For instance, 
(100000000000000000 ( ( Not) ( ( bad))))
has one aspect – overall performance – which is encoded by the first character in the string. Since there are no other aspects, the remaining positions are encoded by ‘0’.
The labelled instances are stored in the train folder in the train, dev and text files.
Implementation Details
The code itself is a modified version of the Tree-Structured LSTM. It departs from the original code in two ways: 
1)	Multi-aspect labels appended only to the root node: This calls for suitable modifications of the loss function. First, loss is only computed for the root node. Second, the ASBA comprises two tasks – aspect detection and polarity prediction for the present aspects. So it makes sense to create two loss heads for each of the tasks in the final layer of the neural network. For task 1, we compute the loss over 18 softmax units and sum them. Each softmax unit in this layer predicts the presence of the corresponding aspect. The loss for task 1 is the sum over losses from each of these units. Similarly for task 2, we compute loss over another set of 18 softmax units. Here, each softmax unit predicts the sentiment class (positive, negative, mildly positive/mildly negative) for the corresponding prospect. For task 2, loss is computed only if the aspect is present otherwise it is taken to be 0. The final loss for task 2 is the sum over all the 18 softmax units. Thus, the final layer has 36 + 54 units.  The final loss is the weighted sum of these two losses where the weights can be set through cross-validation. A key advantage of this approach is that we can adjust the weights depending on which evaluation metric – F1 score for aspect detection or accuracy score for sentiment polarity prediction – matters more to you. 

2)	Non-training of the word vectors: Since the dataset is limited in size, we avoid backpropagating the error into the word2vecs for each word in leaf node of the parse tree.

3)	Feeding inputs to the model – the tree structure can be encoded in many ways. In the original code, 
Requirements
We use the Tensorflow Fold library to facilitate dynamic batching of trees. This allows for a more stable estimate of the gradients which in turn enables a larger learning rate. Combined with faster sweeps through epochs, dynamic batching reduces training time. Note that TensorFlow by itself doesn’t allow for batching of variable sized inputs such as trees.
To use TensorFlow Fold, we used Tenforflow 1.0.0  GPU version on Ubuntu 16.04 with Python 3.6.2. We recommend training over a GPU for speed-up.
We use 300 dimensional word2vecs pre-trained on the Google News corpus as inputs to the leaf nodes. In the current implementation, we store this file in the root folder. After creating a vocabulary over the training, dev and test sets, a word2vec embedding matrix is then created using the Google word2vec file. The out of vocabulary rate for the Latptop training set is ~ 5%. 

Results and Evaluation
For best results, we recommend running the code on a GPU. As part of validation, we track the F1 score for SamEval subtask 1 slot 1 and the accuracy score for subtask 1 slot 3. For our model, both the scores are very competitive with the results of the leading teams in SamEval’15 and SamEval’16.
