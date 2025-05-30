# VGExplainer: A Multi-Granularity Generic Explainer for GNN-Based Vulnerability Detection

## DataSet
The old version of Big-Vul is in [here]( https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing) 

The new version of Big-Vul is in [here](https://drive.google.com/drive/folders/1VPUGYjrhIEXYOdPjYGdwYrHfvGb4LL7O?usp=sharing)

## Requirement
Please check all requirements in the requirement.txt

## Preprocess
Please cheak ```Preprocess``` file folder.

1.Run preprocess file folder  ```raw_data_preprocess.py``` to get codes from big-vul dataset and generate patch lines.
  Or you can use our extracted codes in ```Big_Vul_updated.zip```.

2.Run preprocess/code_normalize file folder ```normalization.py``` to normalize the codes.

3.Use joern to generate PDG graphs, we use v1.1.172, please go to Joern's website: https://github.com/joernio/joern for more details on graph generation.

  We have offered the ```joern script``` file folder.

  We give py scripts in preprocess file folder ```joern_graph_gen.py```.You can refer to the required file.(.bin/.dot/.json)
  
  .bin:Requirement for generating dot graphs and line2node dict.
  
  .dot: the PDG of code
  
  .json: a dict we use to chage node id to line id.
  
4. Use ```slice/main.py``` to generate ground truth and local behaviours.
  
5.Run preprocess file folder ```train_w2v.py``` to get trained w2v model.

6.Run preprocess file folder ```joern_to_devign.py``` to get the data required by the VD model.


## Training of vulnerability detection model
1. All codes in ```vul_detect ``` file folder.
 
2. You need to modify ```data_loader/dataset.py ```.Pay attention to split the training set and test set(like train_set.txt/test.txt to provide data path)
 
3. You need to modify ```main.py ``` and ```trainer.py ``` like some input or output paths.
 
4. Run ```main.py ``` to train or test.

## Run VGExplianer
1. Check ```benchmark/data/dataset.py``` and ```dataset_gen.py``` to appoint and load the graphs you would like to explain.
  
2. Modify ```benchmark/args.py``` about the trained_model_path and explianer args.
   
3. Modify ```benchmark/models/models.py``` about the VD models.
   
4. Modify ```benchmark/models/explainers.py``` to appoint the dot path, slice path and out path.
   
5. run ```benchmark/kernel/pipeline.py```

## Evaluate
1.For node importance calculation, we use a MAX pooling strategy in the paper.

Because we have evaluated that MAX pooling strategy has a better effectiveness than SUM and AVERAGE pooling strategies in ```pooling_strategies/RQ2_supplement.csv``` as a whole.

2.CloneGen is in [here]( https://github.com/CloneGen/CLONEGEN). 

We also provide samples with different transformation and the description table of all 15 transformations in ```CloneGen_sample``` folder.

3.You can use ```Preprocess/evaluate_metrics.py``` to calculate the accuracy, precison, iou and dice coefficient.

4.Except for CVE-2017-6892 in the paper, we also provide vulnerability explanations for VGExplainer in 5 different scenarios to fully demonstrate its effectiveness in the real world:

```other_cases/CVE-2015-8961```:CWE-416

```other_cases/CVE-2017-12179```:CWE-190

```other_cases/CVE-2018-15863```:CWE-476

```other_cases/CVE-2019-15296```:CWE-119

```other_cases/CVE-2020-13910```:CWE-125

We still provide a comparison of DeepSeek-R1:32b's explanation before and after VGExplainer enhancement. We can observe that VGExplainer significantly improves the explanatory performance of DeepSeek-R1, although it is just a small parameter scale model(32b).

5.For LLM evaluation, You can find our scripts and motivation examples in ```LLM``` file folder for reference.

We privide three LLMs for evaluation: GPT-4o, Deepseek-r1, Llama3.1.

For ```GPT_talk.py```, you need to install autogen framework [here](https://github.com/microsoft/autogen) 

For ```deepseek_talk.py``` and ```llama_talk.py```, you need to install Ollama  [here](https://ollama.com/) 
