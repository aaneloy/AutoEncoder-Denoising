# AutoEncoder-Denoising
autoencoder for denoising mnist dataset

## DISCRIPTION
> Denoising autoencoders are an extension of simple autoencoders; however, it’s worth noting that denoising autoencoders were not originally meant to automatically denoise an image. Instead, the denoising autoencoder procedure was invented to help:
>
> * The hidden layers of the autoencoder learn more robust filters
> * Reduce the risk of overfitting in the autoencoder
> * Prevent the autoencoder from learning a simple identify function


## DATASET  
> Later in this tutorial, we’ll be training an autoencoder on the MNIST dataset. The MNIST dataset consists of digits that are 28×28 pixels with a single channel, implying that each digit is represented by 28 x 28 = 784 values.Noise was stochastically (i.e., randomly) added to the input data, and then the autoencoder was trained to recover the original, nonperturbed signal.From an image processing standpoint, we can train an autoencoder to perform automatic image pre-processing for us. 
>
## STRUCTURE of This Project
> the architecture of autoencdoer is in `pyimagesearch/convautoencoder.py` and for starting the train procedure you can run following command:
```
python train_denoising_autoencoder.py
```
furthermore,you can open the `train_denoising_autoencoder.ipynb` in google colab and run it cell by cell,same as below:
> set the matplotlib backend so figures can be saved in the background and import the necessary packages
```
import matplotlib
matplotlib.use("Agg")
from pyimagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
```
> initialize the number of epochs to train for and batch size
```
EPOCHS = 25
BS = 32
```
> load the MNIST dataset
```
print("[INFO] loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()
```
> add a channel dimension to every image in the dataset, then scale the pixel intensities to the range [0, 1]
```
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
```
> sample noise from a random normal distribution centered at 0.5 (since our images lie in the range [0, 1]) and a standard deviation of 0.5)
```
trainNoise = np.random.normal(loc=0.5, scale=0.5, size=trainX.shape)
testNoise = np.random.normal(loc=0.5, scale=0.5, size=testX.shape)
trainXNoisy = np.clip(trainX + trainNoise, 0, 1)
testXNoisy = np.clip(testX + testNoise, 0, 1)
```
> construct our convolutional autoencoder
```
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28, 1)
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)
```
> train the convolutional autoencoder
```
H = autoencoder.fit(trainXNoisy, trainX,validation_data=(testXNoisy, testX),epochs=EPOCHS,batch_size=BS)
```
> construct a plot that plots and saves the training history
```
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('newplot.png')
```
> after running this cell, the result of train/validation basis on our dataset will be creating,such as below :
> 
![plot](https://user-images.githubusercontent.com/53394692/111320859-a6f3e300-867c-11eb-918f-06d4b5d07467.png)
>
> use the convolutional autoencoder to make predictions on the testing images, then initialize our list of output images :
```
print("[INFO] making predictions...")
decoded = autoencoder.predict(testXNoisy)
outputs = None
```
> loop over our number of output samples :
```
for i in range(0,8):
	# grab the original image and reconstructed image
	original = (testXNoisy[i] * 255).astype("uint8")
	recon = (decoded[i] * 255).astype("uint8")

	# stack the original and reconstructed image side-by-side
	output = np.hstack([original, recon])

	# if the outputs array is empty, initialize it as the current
	# side-by-side image display
	if outputs is None:
		outputs = output

	# otherwise, vertically stack the outputs
	else:
		outputs = np.vstack([outputs, output])

# save the outputs image to disk
cv2.imwrite("output.png", outputs)
```
> after run this cell you will be seeing,the two columns,left column has different noisy input images,and in right side you see the output images as denoised of these images as output of autoencoder,such as below :
> 
![output](https://user-images.githubusercontent.com/53394692/111321399-21246780-867d-11eb-907b-5c5bf2d79cd0.png)


## License
> [Denoising autoencoders with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/02/24/denoising-autoencoders-with-keras-tensorflow-and-deep-learning/) by Adrian Rosebrock
