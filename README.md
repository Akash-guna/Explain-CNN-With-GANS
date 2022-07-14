# Repository For the Paper "Interpreting CNNs using Conditional GANs by representing CNNs as Conditional Priors"

The Paper introduced a method to interpret why a CNN makes its predictions by training a GAN to understand CNNs. We propose a novel method that trains conditional Generative Adversarial Networks to generate visual explanations of Convolutional Neural Networks (CNNs). To comprehend a CNN, the GAN is trained with information on how the CNN processes an image when making predictions. Supplying that information has two main challenges. How to represent these information in a form feedable to the GANs and how to effectively feed the representation to the GAN. To solve these issues, we developed a suitable representation of the CNN using cumulatively averaging intermediate explainability maps. We also propose two methods to feed the representations to the GAN and to choose an effective training strategy. Our approach learned the general aspects of CNNs and was agnostic to  datasets and CNN architectures. The study includes both a qualitative and quantitative evaluation of the interpretability maps in comparison with state of the art approaches. We found that the initial layers of CNNs and final layers are equally crucial for explaining CNNs upon interpreting the GAN.
## Our Main Contributions via this work are :
* Introduced a GAN that understands the general working of CNNs.
* Introduced a method to represent CNN's operations as conditional priors.
* Introduced a method to interpret our proposed GAN.

## Proposed Model Architectures
<img src = "assets/model_architecture_cropped.png">


## Inference
Here me mention how to use this repository for Infererence. We provide a pretrained gan model trained on classifiction models to explain Animals-10, Food-11 and CIFAR-10 Datasets. We provide the pretrained model for both LSFT-GAN and GSFT-GAN. We provide a preprocessed .npy file of Food-11 dataset to use for Inference. Please follow the below steps for  Inference:
### Step-1 
Download the npy and pre-trained models from <a href = "https://drive.google.com/drive/folders/1PxKBHLr64gBrLFCi9N_hVNGwZUiYll8f?usp=sharing">Download Drive </a>

### Step-2
Unzip the <b>Models.zip</b> downloaded from drive to models/. <b>Models.zip</b> contains the pretained models for LSFT-GAN and GSFT-GAN
### Step-3
Move the <b>Food_Resnet.npy</b> downloaded from drive to npys/. <b>Food_Resnet.npy</b> is the npy containing the required preprocessed inputs for Inference. 
### Step-4
Create a new python environment and run requirements.txt
In command line:
~~~
py venv -m env
env/scripts/activate.bat
pip install -r requirements.txt
~~~
### Step -5
To Run Inference on LSFT-GAN run
~~~
python lsft_inference.py
~~~

To Run Inference on GSFT-GAN run
~~~
python gsft_inference.py
~~~
### Step -6
Open outputs/ to find the folder with the inferences.
The inference folder contains three sub floders inputs , Gan outputs and GradCAM outputs.
