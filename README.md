# HLIP: A pan-cancer model for histological image analysis in clinical research using TCGA

To install the dependencies, run

<pre/>pip install -r requirements.txt</pre>

The code for all of our experiments is below:

## 1. Zero-shot learning for image classification

<pre/>our_Zshoot.py and our_Zshoot2.py</pre>

## 2. Histology-to-histology retrieval and Clinical feature-to-histology retrieval

<pre/>our_Retrieval.py</pre>

## 3. Histology-based retrieval of cancer stage

<pre/>our_stagePredict.py</pre>

## 4. Histology-based retrieval of survival time

<pre/>our_survive.py</pre>

## 5. High-malignant region prediction

<pre/>analysis_task_5.py</pre>

### Citation

If you find this code to be useful for your research, please consider citing.

<pre>
@article{RONG2025108589,
title = {HLIP: A pan-cancer model for histological image analysis in clinical research using TCGA},
journal = {Computational Biology and Chemistry},
volume = {119},
pages = {108589},
year = {2025},
issn = {1476-9271},
doi = {https://doi.org/10.1016/j.compbiolchem.2025.108589},
url = {https://www.sciencedirect.com/science/article/pii/S1476927125002506},
author = {Jianming Rong and Hengjian Zhong and Yiwei Meng and Qiqi Jin and Yijian Zhang and Chunman Zuo},
keywords = {Multi-Modal Learning, Histology-clinical information alignment, Tumor classification},
abstract = {Whole slide imaging (WSI) captures high-resolution histopathological details, enabling the prediction of clinical outcomes such as tumor classification, disease progression, and patient prognosis. However, existing methods often focus on single-tumor prediction, lack pan-cancer analysis, oversimplify clinical data into categorical labels, and down-sample WSIs, leading to a loss of cellular details, and reduced accuracy. Here, we collected ∼18 K WSIs and 12 clinical features from ∼13 K patients across 32 tumor types in the TCGA database. To enhance histology-clinical associations, WSIs were divided into 512 × 512 patches, and paired with clinical features, generating 190 K histology-clinical feature pairs. We developed histology–language image pretraining (HLIP), a transformer-based model that learns embeddings from clinical paragraph descriptions and histological patches using contrastive learning. HLIP achieved F1@10 (0.886), F1@50 (0.856), and F1@100 (0.915) for zero-shot classification on external datasets, outperforming competing models by 0.6, demonstrating its strong generalization capability. Additionally, HLIP facilitates bidirectional retrieval between histology and 12 clinical features and identifies high-malignant regions in histology, highlighting its strong clinical potential.}
}
</pre>

### Acknowledgement

The implementation of HLIP relies on resources from CLIP. We thank the original authors for their open-sourcing.
