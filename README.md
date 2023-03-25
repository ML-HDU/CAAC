# Character Aware Alignment Contrastive Learning for Chinese Scene Text Recognition

Official PyTorch implementation for Character Aware Alignment Contrastive Learning for Chinese Scene Text Recognition (CAAC).

## Abstract
Indistinguishably caused by enormous character categories and significant character similarity is main challenge for Chinese scene text recognition. In this paper, we explore the topic of discriminative Chinese character representations extraction to cope with the above difficulties. The **C**haracter **A**ware **A**lignment **C**ontrastive learning (CAAC) is proposed for Chinese scene text recognition, which achieves superior performance in an overall concise framework. We leverage the character aware properties of attentional decoder to instantiate character level contrastive learning with more fine-grained atomic elements than previous sub-word level self-supervised contrastive learning based text recognition methods. In addition, the projection-free strategy for directly coupling task loss and supervised contrastive loss is investigated to jointly guide the recognizer to be flexible to Chinese character identification and tradeoff between intra- and inter-domain generalization. All the proposed strategies are plug-and-play, we demonstrate that the CAAC induces stable performance boosting to existing methods and projection-free brings superior cross-domain generalization than three projection heads. Extensive experiments have been conducted on multiple text recognition benchmarks including a self-collected ship license plate dataset to verify the recognition performance, generalization capability and transferability. The experimental results show that our method outperforms previous methods by 2.81% and 1.34% on Chinese ***Scene*** and ***Web*** text recognition datasets.

<img src="./figures/overall_framework.png" width="1000" title="The overall pipeline of the proposed CAAC.">

## Runtime Environment

- Inference requires PyTorch >= 1.7.1
- For training and evaluation, install the dependencies
```
pip install -r requirements.txt
```

## Datasets

- Download ***[Scene](https://github.com/FudanVI/benchmarking-chinese-text-recognition)*** and ***[Web](https://github.com/FudanVI/benchmarking-chinese-text-recognition)*** lmdb dataset for training and evaluation.
- For cross-domain generalization analysis, the commonly used English datasets can be downloaded in [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) (IIIT), [Street View Text](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) (SVT), [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4) (IC15 1811), [Street View Text Perspective](https://mmocr.readthedocs.io/en/v0.6.1/datasets/recog.html#svtp) (SVTP), and [WordArt](https://github.com/xdxie/WordArt).
- The new collected SLPR and SLPR-P dataset consists of 6,922 artistic text images with 5,131 training images and 1,791 testing images. The dataset is available at [Google Drive]().

## Training and Evaluation

- Training

```
python main.py --config=configs/xxx.yaml
```

- Evaluation

```
python main.py --config=configs/xxx.yaml --phase test --image_only
```

Additional flags:

- ``` --test_root /path/to/dataset ``` set the path of evaluation dataset
- ``` --image_only ``` disable dumping visualization of attention masks
- ``` --checkpoint /path/to/checkpoint ``` set the path of pre-trained model
- 


## Acknowledgements

This implementation is based on the repository [ABINet](https://github.com/FangShancheng/ABINet), [SupContrast](https://github.com/HobbitLong/SupContrast), [FudanVI
/benchmarking-chinese-text-recognition](https://github.com/FudanVI/benchmarking-chinese-text-recognition), [WordArt](https://github.com/xdxie/WordArt).


