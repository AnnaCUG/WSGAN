# WSGAN
This is a road network extract method code for remote sensing image dehazing, and this is realized by python. To run this project you need to set up the environment, download the dataset, run a script to process data, and then you can train and test the network models. I will show you step by step to run this project and I hope it is clear enough.

--Prerequisite I tested my project in Intel Core i9, 64G RAM, GPU RTX 2080 Ti. Because it takes about several days for training, I recommend you using CPU/GPU strong enough and about 24G Video Memory.

--Dataset I use a public remote sensing image which consists of 5665 remote sensing images, 5665 mapping images. All the images were 256 × 256 pixels in size.

--Training To train a generator, run the following command python main.py and set the '--phase' as 'train'.

--Test To train a generator, run the following command python main.py and set the '--phase' as 'test'.
