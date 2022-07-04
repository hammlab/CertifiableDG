# On Certifying and Improving Generalization to Unseen Domains

<p align = justify>
Domain Generalization (DG) aims to learn models whose performance remains high on unseen domains encountered at test-time by using data from multiple related source domains. 
Many existing DG algorithms reduce the divergence between source distributions in a representation space to potentially align the unseen domain close to the sources. 
This is motivated by the analysis that explains generalization to unseen domains using distributional distance (such as the Wasserstein distance) to the sources.
However, due to the openness of the DG objective, it is challenging to evaluate DG algorithms comprehensively using a few benchmark datasets.
In particular, we demonstrate that accuracy of the models trained with DG methods varies significantly across unseen domains, generated from popular benchmark datasets.
This highlights that the performance of DG methods on a few benchmark datasets may not be representative of their performance on unseen domains in the wild.
To overcome this roadblock, we propose a universal certification framework based on distributionally robust optimization (DRO) that can efficiently certify the worst-case performance of any DG method. 
This enables a data-independent evaluation of a DG method complementary to the empirical evaluations on benchmark datasets. 
Furthermore, we propose a training algorithm that can be used with any DG method to provably improve their certified performance.
Our empirical evaluation demonstrates the effectiveness of our method at significantly improving the worst-case loss (i.e., reducing the risk of failure of these models in the wild) without incurring a  significant performance drop on benchmark datasets. 
</p>

<hr>

### The codes used to report the results in the paper <b>"[On Certifying and Improving Generalization to Unseen Domains](https://arxiv.org/abs/2206.12364)"</b> are present in this repository.
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

If you find this useful for your work, please consider citing
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
	
