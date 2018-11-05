# Scoring the performance of GANs
This project is meant for evaluating GAN performances (Generative Adversarial Network). 

# Dependency
Make sure to run the project with Python 3.3 or greater.
For those interested in Inception Scores and 1-NN Score with AlexNet, lateset version of tensorflow is need.
# Features Supported
- Scoring GAN performances with these metrics:
    * [Inception Score](https://arxiv.org/pdf/1801.01973.pdf)[to be completed]
    * [1-NN (One Nearset Neighbor) Score](https://arxiv.org/pdf/1802.03446.pdf) [basically done]
    * MAE (Mean Average Error) of pixel-wise color std and mean [basically done]
- Comparing two GANs against a common real distribution [basically done]
- Plotting the evolution of GAN performance as training proceeds [to be implementd]
