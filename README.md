# Zero Shot Learning with Zero Manual Annotations

## Setup
- create a folder named `data` in the root directory of the repo
- download and unzip the following in `data\`:
    - [CUB dataset](https://drive.google.com/file/d/1epi18JVFF4gdVGuLVgjBrrn_1likhf2K/view?usp=sharing)
    - [NAB dataset](https://drive.google.com/file/d/1lpuq9s9Jiz2iOCWJxexYKb8tvW4_kch9/view?usp=sharing)
    - [Wiki Articles](https://drive.google.com/file/d/1oWb8hpg6Ku-pVvBssRaoDTAMJ0c0cRFS/view?usp=sharing)
- run `pip install -r requirements.txt`

## To Run Experiments with TFIDF and PROTO module on the CUB Dataset
Navigate to the main directory (CIS620)
- run `python scripts/Train_CUB_ACE.py` for VPDE features with CE loss
- run `python scripts/Train_CUB_MSE.py` for VPDE features with MSE loss
- run `python scripts/Train_CUB_Res_ACE.py` for ResNet features with CE loss
- run `python scripts/Train_CUB_Res_MSE.py` for ResNET features with CE loss

If you want to run the code without PROTO module (referred to as Baseline in our report) change TransformerFeatures to TransformFeaturesSimple in each of these scripts respectively.

## To Run Experiments with BERT:
- run `python train_bert.py` with desired parameters
- run `python train_bert.py --help` to see possible parameters

