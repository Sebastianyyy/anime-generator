# anime-generator

The Anime Generator is a deep learning model built on the Deep Convolutional Generative Adversarial Network (GAN) architecture. This repository is intended for educational purposes and focuses on generating anime faces. This description covers key aspects such as the dataset, architecture, training process, generation, and references.


# Dataset
The dataset used for this project is available at Kaggle and contains approximately 21,551 anime faces. Each example in the dataset is of size 64x64x3 pixels. It's worth noting that there are some outliers in the dataset, including poorly cropped images and non-human faces. For future work, the goal is to combine multiple datasets containing anime faces to improve the model's results.



# Architecture
The architecture of the Anime Generator is based on the research paper referenced in the last section. The core components of the architecture consist of a discriminator and a generator.

The generator, as illustrated in the diagram below, is composed of convolutional layers and follows the principles outlined in the referenced paper.
![architecture](https://github.com/Sebastianyyy/anime-generator/blob/main/images/architecture.png)


The discriminator, on the other hand, is constructed in a manner almost opposite to the generator. It employs transposed convolutional layers and is designed to distinguish between real and generated images.

Throughout the architecture, Batch Normalization layers and various activation functions such as ReLU, LeakyReLU, and Tanh have been incorporated, following the recommendations from the referenced research paper.




# Training Process
The training process for GANs is challenging, and this project faced additional difficulties due to the absence of GPU access on the local work unit. As a result, attempts were made to train the model using available free GPU resources provided by Kaggle and Google Colab. However, these resources are limited, and as a consequence, the results obtained may not be as optimal as they could be. Despite the limitations, this project serves as a valuable educational experience in GAN-based anime face generation.
I can't provide all the specific hyperparameters used in the project, it's worth mentioning that some of the hyperparameters, such as learning rate and batch size, were adopted from the research papers. Additionally, various hyperparameters, including learning rate, weight decay, and the number of training epochs, were experimented with during the training process. The adaptation of hyperparameters is a common practice in deep learning to optimize model performance.
In the training phase, adjustments were made to the labels of real and fake examples. The values for real examples were experimented with, ranging from 1.0 to 0.9, 0.99, and 0.995, while the values for fake examples were modified from 0.0 to 0.1, 0.01, and 0.005. These adjustments were made iteratively until satisfactory results were achieved, taking into consideration the available GPU resources and the training progress.

# Generation
To generate random anime face, all you need (:D https://arxiv.org/abs/1706.03762) is set number_of_images and execute py generate.py
Generated image has size of 64x64x3 , same as training examples. Not all generated examples are great, it is caused of not best trained model.
# Generated example
![anime-1-example](https://github.com/Sebastianyyy/anime-generator/blob/main/images/anime.png)

# Generated 48 examples
![anime-48-example](https://github.com/Sebastianyyy/anime-generator/blob/main/images/anime1.png)



# references
https://arxiv.org/abs/1511.06434
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

ReadMe was created with the help of chatgpt.