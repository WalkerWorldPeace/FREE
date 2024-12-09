# [CVPR 2024] FREE: Faster and Better Data-Free Meta-Learning

## Abstract
Data-Free Meta-Learning (DFML) aims to extract knowledge from a collection of pre-trained models without requiring the original data presenting practical benefits in contexts constrained by data privacy concerns. Current DFML methods primarily focus on the data recovery from these pre-trained models. However they suffer from slow recovery speed and overlook gaps inherent in heterogeneous pre-trained models. In response to these challenges we introduce the Faster and Better Data-Free Meta-Learning (FREE) framework which contains: (i) a meta-generator for rapidly recovering training tasks from pre-trained models; and (ii) a meta-learner for generalizing to new unseen tasks. Specifically within the module Faster Inversion via Meta-Generator each pre-trained model is perceived as a distinct task. The meta-generator can rapidly adapt to a specific task in just five steps significantly accelerating the data recovery. Furthermore we propose Better Generalization via Meta-Learner and introduce an implicit gradient alignment algorithm to optimize the meta-learner. This is achieved as aligned gradient directions alleviate potential conflicts among tasks from heterogeneous pre-trained models. Empirical experiments on multiple benchmarks affirm the superiority of our approach marking a notable speed-up (20x) and performance enhancement (1.42%~4.78%) in comparison to the state-of-the-art.
## Requirements

```
pip install -r requirements.txt
```

## Datasets & Pre-trained Modes:

**Datasets:**

* **CIFAR-FS:** 

  * Please manually download the CIFAR-FS dataset.

  * Unzip ".zip". The directory structure is presented as follows:

    ```css
    cifar100
    ├─mete_train
    	├─apple (label_directory)
    		└─ ***.png (image_file)
    	...
    ├─mete_val
    	├─ ...
    		├─ ...
    └─mete_test
    	├─ ...
    		├─ ...
    ```

  * Place it in "./DFL2Ldata/".

* **Mini-Imagenet:** Please manually download it. Unzip and then place it in "./DFL2Ldata/".

* **CUB:** Please manually download it. Unzip and then place it in "./DFL2Ldata/".

**Pre-trained models:**

- You can pre-train the models following the instructions below (Step 3).

## Training:

1. Make sure that the root directory is "./FREE".

2. Prepare the dataset files.

   - For CIFAR-FS:

     ```shell
     python write_file/write_cifar100_filelist.py
     ```

     After running, you will obtain "meta_train.csv", "meta_val.csv", and "meta_test.csv" files under "./DFL2Ldata/cifar100/split/".

   - For MiniImageNet:
     ```shell
     python write_file/write_miniimagenet_filelist.py
     ```
     
   - For CUB:
     ```shell
     python write_file/write_CUB_filelist.py
     ```
    
3. Prepare the pre-trained models.

    ```shell
    bash ./scripts/pretrain.sh
    ```
	
    Some options you may change:

    |     Option     |           Help            |
    | :------------: | :-----------------------: |
    |   --dataset    | cifar100/miniimagenet/cub |

4. Data-free meta-learning
   - For main results:
     ```shell
      bash ./scripts/mr.sh
     ```
   - For multi-domain scenario:
     ```shell
      bash ./scripts/md.sh
     ```
   - For multi-architecture scenario:
     ```shell
      bash ./scripts/ma.sh
     ```
   Some options you may change:
   
   |     Option     |           Help            |
   | :------------: | :-----------------------: |
   |   --dataset    | cifar100/miniimagenet/cub/mix |
   | --num_sup_train |  1 for 1-shot, 5 for 5-shot  |

## Citation
If you find FREE useful for your research and applications, please cite using this BibTeX:
```bash
@inproceedings{wei2024free,
  title={{FREE}: Faster and Better Data-Free Meta-Learning},
  author={Wei, Yongxian and Hu, Zixuan and Wang, Zhenyi and Shen, Li and Yuan, Chun and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
## Acknowledgement
Some codes are inspired from [Fast](https://github.com/zju-vipa/Fast-Datafree) and [BiDf-MKD
](https://github.com/Egg-Hu/BiDf-MKD).

