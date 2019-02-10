# General

In this work we  use Dreem headband data to perform sleep stage scoring on 30 seconds epochs of biophysiological signals.
This work is  a part of an in-class [Kaggle Competition](https://www.kaggle.com/c/dreem-sleep-stages) for the Machine
Learning Course offered at Ecole CentraleSupelec, Paris in the Fall 2018-2019.



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
 |————————-|————-|—————|————-|—————|
 |Min - max - freq - energy on pulse et accelerometre|0.518055 |200 |0.563882 |150 |
 |Min - max - freq - energy on eegs |0.563267 |300 |0.625435 |300 |
 |Min - max - freq - energy on all|0.589399 |400 |0.639735 |400 |
 |Min - max - freq - energy on all + Riemann|0.604606 |2500 |0.639785 |2000 |
 |Min - max - freq - energy on all + CSP |0.617818 |3000 |0.639209 |3000 |

 ## Further work :

 Markup : * A good Idea would be to a convnet architecture on the Riemann or CSP features to finish the classification
 instead of LGBM method

          * Dindin Meryll has done a really impressive work on this problematic. He achieved to have a 70.7 % accuracy.
            He explains his work in [this blog](https://towardsdatascience.com/my-sweet-dreams-about-automatic-sleep-stage-classification-414128441728?fbclid=IwAR0ROZvlMFr1NY2wmtF-xqlOrpYbbxgUFXo_iILtnQLuhfP6ACM2xIlZFrA).

          * A recent paper : XXX can also be a very good start for an inference approach.



