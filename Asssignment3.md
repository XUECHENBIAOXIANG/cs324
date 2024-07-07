#                                      Assignment3

## Student Name: biaoxiang chen

## Student ID:12112202

##  Introduction

In this assignment, we delve deeper into the world of `Recurrent Neural Networks` (`RNNs`) by implementing a Long Short-Term Memory (`LSTM`) network from scratch using `PyTorch`. Additionally, we explore the realm of `Generative Adversarial Networks` (`GANs`), specifically focusing on generating images akin to those found in the `MNIST` dataset.

## Part I: PyTorch LSTM

### Task 1: Implementing LSTM

In this task, we implemented an `LSTM` model without utilizing the built-in `torch.nn.LSTM` module.  The weight matrices` Wgx`, `Whx`, `Wix`, `Whi`, `Wfx`, `Whf`,` Wox`, `Who`,` Wph` and bp are initialized with `nn.Parameter`, which means that they are automatically updated by the optimizer.These are the building blocks of `lstm`'s four main gating units, including the Forget gate, the input gate, the unit status gate, and the output gate. The initialization of the weight matrix has great influence on the performance of the model. Here, uniformly distributed random initialization is used, a common practice that helps break symmetry and promotes better training. we use` torch.sigmoid`，`torch.tanh` and`softmax` activation to  generate candidate cell states ，to complete forwarding.

### Task 2: Palindrome Prediction

We tested the `LSTM` model with varying input sequence lengths (T=5, 10, 19,25,30 and 35). The model demonstrated improved accuracy and performance compared to the` RNN` model implemented in the previous assignment. Notably, the` LSTM `model achieved near-perfect accuracy for T=5 with the default parameters provided.

####  the specific test parameters

| Config | Input Length | Input Dim | Num Classes | Num Hidden | Batch Size | Learning Rate | Max Epoch | Max Norm | Data Size | Portion Train |
| ------ | ------------ | --------- | ----------- | ---------- | ---------- | ------------- | --------- | -------- | --------- | ------------- |
| 1      | 5            | 1         | 10          | 128        | 100        | 0.001         | 100       | 10.0     | 100000    | 0.8           |
| 2      | 10           | 1         | 10          | 128        | 100        | 0.001         | 100       | 10.0     | 100000    | 0.8           |
| 3      | 19           | 1         | 10          | 128        | 100        | 0.001         | 100       | 10.0     | 100000    | 0.8           |
| 4      | 25           | 1         | 10          | 128        | 100        | 0.001         | 100       | 10.0     | 100000    | 0.8           |
| 5      | 30           | 1         | 10          | 128        | 100        | 0.001         | 100       | 10.0     | 100000    | 0.8           |
| 6      | 35           | 1         | 10          | 128        | 100        | 0.001         | 100       | 10.0     | 100000    | 0.8           |

#### the running results

**Input Length =5**

<img src="C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523104159903.png" alt="image-20240523104159903" style="zoom: 67%;" />

**Input Length =10**

<img src="C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523104235379.png" alt="image-20240523104235379" style="zoom:67%;" />

**Input Length =19**

<img src="C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523104302239.png" alt="image-20240523104302239" style="zoom:67%;" />

**Input Length =25**

<img src="C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523104421061.png" alt="image-20240523104421061" style="zoom:67%;" />

**Input Length =30**

<img src="C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523104438895.png" alt="image-20240523104438895" style="zoom:67%;" />

<img src="C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523104516351.png" alt="image-20240523104516351" style="zoom:67%;" />

#### Performance Analysis

Here is a detailed analysis of the performance of the` LSTM` model based on the provided configurations and observations:

##### Short Sequence Lengths

For shorter sequence lengths, the `LSTM` model performs remarkably well, achieving stable accuracy  near 100%. This indicates that for tasks with shorter dependencies, the `LSTM's` ability to retain and process information is highly effective.

##### Longer Sequence Lengths

As the sequence length increases, the `LSTM` model faces challenges that require adjustments to the training parameters. 

A higher learning rate is necessary to speed up learning when dealing with longer sequences. This helps the model adjust weights more quickly, adapting to the complexities of longer dependencies.

##### other

**Comparative Performance**: Compared to the `RNN` model used in a previous task, the `LSTM` shows superior performance. `LSTMs` are specifically designed to handle long-term dependencies better than traditional `RNNs`.

**Long-Term Dependency Issue**: Despite being designed to address long-term dependencies, `LSTMs` can still struggle with very long sequences. The memory cells might prematurely forget information or decay during propagation, making it challenging to learn long-term dependencies effectively. Alternative architectures or additional techniques (e.g., `attention mechanisms`) might be necessary to effectively capture long-term dependencies.

**Something strange**:<font color=#FF0000 > **The test set works better than the training set.**</font>There is a high probability that a `data breach` has occurred. Because I made changes to the code that generated the data.

```python
self.data = np.random.default_rng().choice(
    max_num, self.total_len, replace=True)
```

with replace is true, allows for repeated values in the sampled data,so Perhaps the test set is directly too similar to parts of the training set or the distribution is not consistent, the test set is simple and the training set is difficult.

## Part II: Generative Adversarial Networks

### Task 1: Building the GAN

We constructed our `GAN` using `PyTorch`, with the generator and discriminator networks designed to produce and distinguish `MNIST`-like images, respectively. The generator consists of multiple layers with `LeakyReLU` activations, while the discriminator employs a similar structure to classify images as real or fake. Below is a detailed description of the architecture and training process.

#### Generator Network

The generator network takes a latent vector as input and produces a 28x28 image. The architecture consists of multiple layers with` LeakyReLU` activations, `batch normalization`, and ` Tanh` activation at the output to normalize the pixel values.

- Architecture

  - Input: Latent vector of size `latent_dim`

  - Fully connected layer to 128 units

  - Batch normalization

  - LeakyReLU activation (negative slope 0.2)

  - Fully connected layer to 256 units

  - Batch normalization

  - LeakyReLU activation

  - Fully connected layer to 512 units

  - Batch normalization

  - LeakyReLU activation

  - Fully connected layer to 512 units

  - Batch normalization

  - GELU activation

  - Fully connected layer to 1024 units

  - Batch normalization

  - LeakyReLU activation

  - Fully connected layer to 784 units
  
  - Tanh activation to normalize pixel values between -1 and 1

#### Discriminator Network

The discriminator network takes a 28x28 image as input and produces a probability indicating whether the image is real or fake. The architecture consists of multiple fully connected layers with `LeakyReLU` activations and a `Sigmoid activation` at the output.

- Architecture:
  - Input: Flattened image of size 784 (28x28)
  - Fully connected layer to 512 units
  - LeakyReLU activation
  - Fully connected layer to 256 units
  - LeakyReLU activation
  - Fully connected layer to 1 unit
  - Sigmoid activation to produce a probability
#### Training Process
The training process involves alternating between updating the discriminator and the generator. The discriminator is trained to distinguish between real and fake images, while the generator is trained to produce images that can fool the discriminator.

- Loss Function: Binary Cross-Entropy Loss (`BCELoss`)

- Optimizers: Adam optimizers for both the generator and the discriminator

- Training Loop:
  
  For each epoch and batch:

  - Train the discriminator on real and fake images
  
  - Train the generator to produce images that the discriminator classifies as real
  
  - Save generated images at specified intervals
  

#### GAN Training Parameters

| Parameter       | Description                            | Default Value |
|-----------------|----------------------------------------|---------------|
| `--n_epochs`    | Number of epochs                       | 200           |
| `--batch_size`  | Batch size                             | 64            |
| `--lr`          | Learning rate                          | 0.0002        |
| `--latent_dim`  | Dimensionality of the latent space     | 100           |
| `--save_interval` | Save model every SAVE_INTERVAL iterations | 500           |


### Task 2: Sampling Images

 At various stages of training (beginning, halfway, and end), we sampled 25 images from our trained `GAN`. These images were documented in a Jupyter notebook, showcasing the progression and improvement in image quality over the course of training.

**start：**

![image-20240523222144856](C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523222144856.png)

**halfway：**

![image-20240523222252528](C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523222252528.png)

**finish：**

![image-20240523222309917](C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523222309917.png)

At the beginning, the effect of sampling generation is very bad, close to noise. In the half of training, the model generation effect is not bad, and it can be obviously observed that digital pictures can be generated, but it is still not clear. At the end of training, it can be observed that very clear digital pictures are basically consistent with the` mnist` data set.

### Task 3: Latent Space Interpolation

We generated two distinct images from our `GAN `and interpolated between them in the latent space. By gradually increasing the percentage of the second noise vector from 0 to 1, we produced a sequence of 9 images that transitioned from the initial to the final image.

![image-20240523222328426](C:\Users\aoxiangxueyuan\AppData\Roaming\Typora\typora-user-images\image-20240523222328426.png)

The process of change can be clearly observed from the picture above.

## Results and Analysis

### LSTM Performance 

The `LSTM `model showed a significant improvement in handling long sequences compared to traditional` RNNs`. However, challenges  long-term dependency issues were still present for very long sequences. 

### GAN Image Quality

The adversarial training process led to a noticeable improvement in the quality of generated images. Initially, the images resembled random noise, but by the end of training, they were clear, consistent, and nearly indistinguishable from real` MNIST `images. 

### Latent Space Interpolation

The interpolation experiment demonstrated the generator's ability to smoothly transition between different image representations in the latent space, showcasing the flexibility and power of` GANs`. 

## Conclusion

This assignment provided valuable insights into the implementation and application of` LSTMs `and `GANs`. The `LSTM `model's enhanced capability to handle long sequences and the `GAN's `ability to generate realistic images highlight the potential of these neural network architectures in various domains.

## Points that can be improved

The effects of different parameters of` lstm` and the generation of data sets are limited by 35, and the `batch_size` is directly linked to the architecture, whether it can be decouple, and the effect experience of `cgan.`
