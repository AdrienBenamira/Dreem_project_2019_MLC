# General

In this work we  use Dreem headband data to perform sleep stage scoring on 30 seconds epochs of biophysiological signals.
This work is  a part of an in-class [Kaggle Competition](https://www.kaggle.com/c/dreem-sleep-stages) for the Machine
Learning Course offered at Ecole CentraleSupelec, Paris in the Fall 2018-2019.

## Challenge goals

Perform sleep stage scoring accurately.

## Data description

Each sample represents 30 seconds of recording for a size total dimension of 22500. There are three kinds of electrophysiological signals: electroencephalogram, pulse oximeter, accelerometer, leading to the following structure of samples in the dataset:

    4 EEG channels sampled at 125Hz (4 x 125 x 30 = 15000 size per sample)
    2 Pulse Oxymeter channels (red and infra-red) sampled at 50 Hz (2 x 50 x 30 = 3000 size per sample)
    3 Accelerometer channels sampled at 50Hz (3 x 50 x 30 = 4500 size per sample)

## Output Description:

Integer between 0 and 4 representing the sleep stage of the 30-second window. The sleep classification follows the AASM recommendations and was labeled by a single expert.

    0 : Wake
    1 : N1 (light sleep)
    2 : N2
    3 : N3 (deep sleep)
    4 : REM (paradoxal sleep)

## Datasets description:

43830 train samples for 20592 test samples

## Our results

Our final F-score is 64.78 % on the private test set and we are currently ranked 4th / 74.


# Install
Create a `builds` directory.
Create a `dataset` directory and add the files `test.h5`, `train.h5` and `train_y.csv`.
To get the results, run first generate_data in tutorials then `Run_upload_final.ipynb`

# Results and report

Report is available [here](https://github.com/AdrienBenamira/Dreem_project_2019_MLC/blob/master/Rapport.ipynb)



|Modèle et paramètres | Résultat de F1-score moyen sur 80% du dataset en CV | Résultat en F1-score en soumission |
| ------------- |:-------------:| -----:|
|Random Forest, paramètres par défaut | 0.61330 | 0.64223 |
|Random Forest, paramètres optimisés | 0.62951 | 0.64303 |
|Random Forest OneVSall | 0.61234 | 0.59698 |
|Gradient Boosting, paramètres par défaut| 0.64723 | 0.63320 |
|Gradient Boosting, paramètres optimisés | 0.66147 | 0.64410 |
|LightGBM | 0.67140 | 0.6478 |



# Extra :

## Feature Comparison :

Différents models | F1-score val sur tout le dataset | Temps d'exécution (s) |F1-score val sur un subdataset balanced | Temps d'exécution (s) |
| ------------- |:-------------:| -----:|------------- |:-------------:|
 |Min - max - freq - energy on pulse et accelerometre|0.518055 |200 |0.563882 |150 |
 |Min - max - freq - energy on eegs |0.563267 |300 |0.625435 |300 |
 |Min - max - freq - energy on all|0.589399 |400 |0.639735 |400 |
 |Min - max - freq - energy on all + Riemann|0.604606 |2500 |0.639785 |2000 |
 |Min - max - freq - energy on all + CSP |0.617818 |3000 |0.639209 |3000 |

 ## Further perspectives :

 * A good Idea would be to a convnet architecture on the Riemann or CSP features to finish the classification
 instead of LGBM method
 * Dindin Meryll has done a really impressive work on this problematic. He achieved to have a 70.7 % accuracy.
 He explains his work in [this blog](https://towardsdatascience.com/my-sweet-dreams-about-automatic-sleep-stage-classification-414128441728?fbclid=IwAR0ROZvlMFr1NY2wmtF-xqlOrpYbbxgUFXo_iILtnQLuhfP6ACM2xIlZFrA).

  * The interesting [BagNet paper](https://openreview.net/forum?id=SkfMWhAqYQ) (accepted at ICLR 2019)
  can also be a very good start for an inference approach.



