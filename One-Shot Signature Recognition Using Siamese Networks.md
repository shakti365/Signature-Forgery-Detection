# One-Shot Signature Recognition Using Siamese Networks

## Solution Design

### Definitions:

- **System**: Herein referred to the signature recognition system to be built
- **User**: entity who will use this system for signature recognition
- **Customer**: entity whose signatures are to be verified
- **Signature**: hand written signature of a Customer
- **Match Score**: A probability value output from the system in range [0, 1], where 1 indicates match and 0 indicates mis-match of signature

### Description of the Solution:

This solution is intended to provide its users with a Signature Recognition system. The user can upload a new customer's signature in the system corresponding to its UUID. Then, a user can upload signature to match the one previously stored in database for the customer. To this, system would be able to provide match score.

#### Inputs:

- **customer_id**: UUID of customer whose signature is to be matched
- **signature**: image of the handwritten signature of a customer

#### Outputs:

- **match_score**: predicted probabillity of a match

#### Components:

- **Frontend**
  - Dashboard to perform actions
- **Backend**
  - APIs for storing genuine signature and matching
  - Manage control flow and communications from different modules
- **Database**
  - Store processed data of customer signatures
- **Model Server**
  - Serve tensorflow models over a REST API
- **Training module**
  - Relevant functions for training and tuning models
- **Inference module**
  - Input and output management from model server
- **Data processing module**
  - Clean and pre-process input data for model and storing in DB

#### Activity Diagram:

![signature-recognition](https://i.imgur.com/4kbye9S.png)

## Tech Stack 

- Programming Langugage: Python
- Frameworks:

  - Tensorflow, Flask


## Algorithms/Models

### Siamese Neural Network

![siamese-model](https://i.imgur.com/lwRkFYF.png)

We will be using a Convolutional Siamese Neural Network model for signature verifications. The model architecture is proposed as SigNet [1]. This model is analogous to the SOTA models used for face verfication tasks [2]. The main goal of this model is to erradicate the task of manual feature engineering for singatures and leverage the representation learning capability of deep neural network architectures. In addition to this Siamese Networks can help in learning with less number of samples or one-shot learning. One-shot learning capability is useful in our case where we want to learn from usually one signature of a customer. Siamese Networks take in two input $x_1, x_2$ and output embeddings $s_1, s_2$ for each of the two inputs using a model parametrized by the same weights $\theta$.
$$
s_1, s_2 = f(x_1, x_2;\theta)
$$
These embeddings are then passed to logistic regression layer which tries to learn a distance measure in the embedding space. Learning distance measure is easier for the model than learning a classification to all the customers singature available.
$$
D_w = ||f(s_1;w_1) - f(s_2;w_2)||
$$
This model then tries to minimize a *contrastive loss* $L(s_1, s_2, y)$ which brings output embeddings of same inputs closer together. Where $y$ is the binary label indicating inputs are similar or different.
$$
L(s_1, s_2, y) = \alpha(1-y)D_{w}^2 + \beta.y .max(0, m-D_w)
$$
We do not go into the details of this loss function here because we want to compare it with a different loss function. Albeit this loss function works well in most of cases but we propose to compare it against using *triplet loss* [3].

### Triplet loss function

The constrastive loss tries to bring embeddings of same inputs closer together. This might seem greedy in a sense that it doesn't account for difference in inputs. Training with triplet loss function mitigates this issue, it brings similar input close together and at the same time tries to separate the inputs farther together in embedding space.

 ![triplet-loss](https://i.imgur.com/QyIWduV.png)

The approach that we propose here is a one-shot siamese network with triplet loss. The goal will be to train the model using data set provided in order to learn the distance measure. The model will then be served using tensorflow and flask backend for match score predictions. Our aim with this approach is to reduce the error rate to a minimum, however, we realise that it might not be possible to reach zero error rates and best use cases of AI comes only when humans can understand and work together with such system. In order to support the end users, we would also like to propose a system where end users can understand why a given match score is predicted by the model using model interpretability techniques [4, 5]. This can serve as a feedback mechanism and process for quality assurance of the system. 

### Results

The data set provided was split in three parts comprising 80% of train set, 10% of validatiion set and 10% as held out test set. The model was trained and on the train set using GPU runtime in colab notebooks. Hyperameter tuning was done by observing metrics like loss, accuracy, precision and recall on the validation set. Once the model achieved desired state it was tested out on the held out data set, where following results were observed.

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.96  |
| Precision | 0.925 |
| Recall    | 1.0   |

A more robust test set with noisy background image in real environment is planed and will be tested in next phase.

### References:

1. Dey et.al. *SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification* https://arxiv.org/abs/1707.02131

2. Taigman et.al. *DeepFace: Closing the Gap to Human-Level Performance in Face Verification* https://research.fb.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/

3. Schroff et.al. *FaceNet: A Unified Embedding for Face Recognition and Clustering* https://arxiv.org/abs/1503.03832

4. Shrikumar et.al. *Learning Important Features Through Propagating Activation Differences* https://arxiv.org/abs/1704.02685

5. Ribeiro et.al *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* https://arxiv.org/abs/1602.04938
