# On Certifying and Improving Generalization to Unseen Domains

<p align = justify>
Domain Generalization (DG) methods use data from multiple related source domains to learn models whose performance does not degrade on unseen domains
at test time. Many DG algorithms rely on reducing the divergence between the
source distributions in a representation space to potentially align unseen domains
close to the sources. These algorithms are motivated by the analytical works that
explain generalization to unseen domains based on their distributional distance
(e.g., Wasserstein distance) to the sources. However, we show that the accuracy of
a DG model varies significantly on unseen domains equidistant from the sources in
the learned representation space. This makes it hard to gauge the generalization
performance of DG models only based on their performance on benchmark datasets.
Thus, we study the worst-case loss of a DG model at a particular distance from the
sources and propose an evaluation methodology based on distributionally robust
optimization that efficiently computes the worst-case loss on all distributions within
a Wasserstein ball around the sources. Our results show that models trained with
popular DG methods incur a high worst-case loss even close to the sources which
show their lack of generalization to unseen domains. Moreover, we observe a
large gap between the worst-case and the empirical losses of distributions at the
same distance, showing the performance of the DG models on benchmark datasets
is not representative of their performance on unseen domains. Thus, our (target) data-independent and worst-case loss-based methodology highlights the poor
generalization performance of current DG models and provides insights beyond
empirical evaluation on benchmark datasets for improving these models. 
</p>

<hr>

### The codes used to report the results in the paper <b>"[Do Domain Generalization Methods Generalize Well?](https://openreview.net/pdf?id=SRWIQ0Yl53m)"</b> are present in this repository.
<hr>


### Obtaining the data:
<ul>
<li> For R-MNIST, we create the dataset by modifying the standard MNIST dataset using the code provided in the following repository https://github.com/facebookresearch/DomainBed. After creating data for different domains based on the rotation angles, we randomly select 1000 points for training and 1000 points for testing.
<li> For VLCS and PACS we download the data using the code from https://github.com/facebookresearch/DomainBed.
</ul>
	

	
### Training models with Vanilla DG methods
Each folder contains the codes for different domain genralization (DG) algorithms we have used in this paper. The folders are split according to the dataset.

Each folder contains files with "vanilla" in their names. These are codes for training vanilla DG algorithms and we provide codes for Wasserstein Matching (WM), G2DM, CDAN and VREX that we used for our paper. These only require an optional TARGET argument (for debugging) which helps evaluate the performnace of a model on an unseen domain.

A sample command to run an experiment is <code> python vanilla_wm.py </code>

### Training models with DR-DG method 
The files with names containing "dr_dg" are the codes for training models with our DR-DG algorithm described in Alg. 2 of the paper. These codes require an argument FACTOR to run. This argument can be any value but in the experiments we find using values between in [0.1, 0.5] works the best for most methods without the need for tuning any other hyperparameters such as the learning rates. 

A sample command to run an experiment is <code> python dr_dg_wm.py --FACTOR 0.25 </code>

### Certifying Vanilla and DR-DG trained models
The models trained with "vanilla" and "dr_dg" can be certified using "cert_dg.py" as described below.

Each folder contains a file named cert_dg.py which is used for certification of differnt models using our Cert-DG algorithm (Alg. 1 of the paper). The code requires three main and one optional argument.
	<ul>
	<li> NAME: which is used to indicate the name of the DG algorithm and can be one of WM, G2DM, CDAN or VREX.</li>
	<li> METHOD: which is used to indicate whether Vanilla trained model needs to be certified or a model trained with DR-DG. Changing the variable METHOD in the code allows for this. Set METHOD = "vanilla_dg" for vanilla models or METHOD = "rep_dro_dg" for models trained with DR-DG.</li>
	<li> FACTOR: For models trained with DR-DG, provide the FACTOR that was used during training. </li>
	<li> TARGET: (only used for debugging) To see performance of the models on a particular unseen domain supply the TARGET variable. </li>
	</ul>
	
A sample command to run an experiment is <code> python cert_dg.py --NAME WM --FACTOR 0.5 </code> (for a DR_DG model with variable METHOD set to "rep_dro_dg")
  
#### Citing

If you find this repository useful for your work, please consider citing our work.
<pre>
<code>
@misc{mehra2022certifying,
      title={On Certifying and Improving Generalization to Unseen Domains}, 
      author={Akshay Mehra and Bhavya Kailkhura and Pin-Yu Chen and Jihun Hamm},
      year={2022},
      eprint={2206.12364},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</code>
</pre>
	
