This folder contains the notebooks and results presented in the paper
* [results](results): Results for different BO runs for each protein dataset
* [Different_featurizations.ipynb](Different_featurizations.ipynb): Notebook for the other two encoding apporach to obtain results in Table 4
* [Example_study_GB1_2016.ipynb](Example_study_GB1_2016.ipynb): notebook to generate results for GB1 (4) dataset from Wu et al (2016) in Table3, Figure 4, Figure 5B and Figure 6
* [ODBO_for_different_datasets.ipynb](ODBO_for_different_datasets.ipynb): notebook to generate results for other three datasets
* [Plot_results.ipynb](Plot_results.ipynb): Notebook to systematiclly plot results and compute statstics
* [Update_XGBOD_GB1_2016.ipynb](Update_XGBOD_GB1_2016.ipynb): Notebook that performs XBGOD every BO iteration instead of before BO procedure in Figure S1
* [XGBOD_accuracy_comparison_GB1_2016.ipynb](XGBOD_accuracy_comparison_GB1_2016.ipynb): Notebook to analyze the XGBOD accuracy with different thresholds (Table 2)

The following ```.npy``` files store the encoding information for
* Physicochemical encoding: sequences in [example_protein_georgiev_IndexToCombo.npy](example_protein_georgiev_IndexToCombo.npy) and feature vectors in [example_protein_georgiev_Normalized.npy](example_protein_georgiev_Normalized.npy)
* One-hot encoding: sequences in [example_protein_onehot_IndexToCombo.npy](example_protein_onehot_IndexToCombo.npy) and feature vectors in [example_protein_onehot_Normalized.npy](example_protein_onehot_UnNormalized.npy)

The following ```.npy``` files store the initial selected experiment information
* **GB1 (4)**
  - 40 initial samples selected by Algorithm S2, sequence in [sele_experiment_GB1_2016.npy](sele_experiment_GB1_2016.npy), fitness in [sele_fitness_GB1_2016.npy](sele_fitness_GB1_2016.npy)
  - 40 randomly selected samples, sequence locations in [sele_indices_GB1_2016_random1.npy](sele_indices_GB1_2016_random1.npy), [sele_indices_GB1_2016_random2.npy](sele_indices_GB1_2016_random2.npy), [sele_indices_GB1_2016_random3.npy](sele_indices_GB1_2016_random3.npy), 
  [sele_indices_GB1_2016_random4.npy](sele_indices_GB1_2016_random4.npy), [sele_indices_GB1_2016_random5.npy](sele_indices_GB1_2016_random5.npy); shuffle orders in [shuffle_order_GB1_2016_random1.npy](shuffle_order_GB1_2016_random1.npy),
  [shuffle_order_GB1_2016_random2.npy](shuffle_order_GB1_2016_random2.npy), [shuffle_order_GB1_2016_random3.npy](shuffle_order_GB1_2016_random3.npy), [shuffle_order_GB1_2016_random4.npy](shuffle_order_GB1_2016_random4.npy), [shuffle_order_GB1_2016_random5.npy](shuffle_order_GB1_2016_random5.npy)
* **GB1 (55)** 
  - 40 initial samples selected by Algorithm S3, sequence locations in [sele_indices_GB1_2014.npy](sele_indices_GB1_2014.npy); shuffle orders in [shuffle_order_GB1_2014.npy](shuffle_order_GB1_2014.npy)
* **Ube4b**
  - 40 initial samples selected by Algorithm S3, sequence locations in [sele_indices_Ube4b_2013.npy](sele_indices_Ube4b_2013.npy); shuffle orders in [shuffle_order_Ube4b_2013.npy](shuffle_order_Ube4b_2013.npy)
* **avGFP**
  - 40 initial samples selected by Algorithm S3, sequence locations in [sele_indices_avGFP_2016.npy](sele_indices_avGFP_2016.npy); shuffle orders in [shuffle_order_avGFP_2016.npy](shuffle_order_avGFP_2016.npy)
