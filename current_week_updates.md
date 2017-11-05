---

## 2017-11-04

Using lipreader as an information retrieval system

![lrw_wordDuration_dense_softmax_critic](20171104/lrw_wordDuration_dense_softmax_critic.png "lrw_wordDuration_dense_softmax_critic")

### mean Average Precisions @ K

![Lipreader vs <Lipreader minus critic_rejects>](20171104/APs_at_K_vs_K_with_logReg_critic_test.png "APs_at_K_vs_K_with_logReg_critic_test")

Figure 1: mean Average Precisions @ K vs K, on LRW (test)

### Lipreader vs "Lipreader minus critic_rejects"

![Lipreader vs <Lipreader minus critic_rejects>](20171104/AP_at_K_vs_word_gray_test.gif "AP_at_K_vs_word_gray_test")

Figure 2: Average Precision for every word (a) using lipreader, (b) using lipreader and rejecting those predicted by critic to be false

### CONCLUSIONS

- mAP: Better mAP for "Lipreader minus critic"

- AP vs word: better AP for most words, worse AP for some words

### TO DO

- Extract head pose for LRW (need fusor!)

- Better critic with LSTM


### SUMMARY

- Read about Multi-class classification metrics [1-4]
    - ROC AUC - not good, PR - better, ROC VUC - maybe

- GRIDcorpus:
    - Lipreader (mAP = 0.98), C3DCritic (mAP = 0.97)

- LRW information retrieval
    - mAP: 0.78; recall = 0.7

- Updated website: http://preon.iiit.ac.in/~vikram_voleti/weekly_updates.html
    - C3DCritic is overfit! - need to train an assessor

- Average Precision @K [5]

- Visualized lipreader average precisions @K

- Trained Logistic Regression critic with word\_durations, lipreader\_dense, lipreader\_softmax
    - 1) unoptimized, weight-unbalanced, threshold = 0.7 is closer to (0, 1) in ROC

        [ROC - logReg\_critic\_unbalanced](20170411/logReg_critic_unbalanced.png)

    - 2) unoptimized, weight-balanced, threshold = 0.5 - not much change

- Visualized average precisions @K after rejects by critic, compared with lipreader's

- Compared mAP@K


[1] Song, Bowen et al. “ROC Operating Point Selection for Classification of Imbalanced Data with Application to Computer-Aided Polyp Detection in CT Colonography.” International journal of computer assisted radiology and surgery 9.1 (2014): 79–89. PMC. Web. 28 Oct. 2017. [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3835757)

[2] Vincent Van Asch. “Macro- and micro-averaged evaluation measures” [link](http://www.cnts.ua.ac.be/%7Evincent/pdf/microaverage.pdf)

[3] D. Hand, R. Till. “A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems,” Machine Learning, 45, 171–186, 2001[link](https://link.springer.com/content/pdf/10.1023%2FA%3A1010920819831.pdf)

[4] E. Fieldsend, R. Everson, “Visualisation of multi-class ROC surfaces” [link](http://users.dsic.upv.es/~flip/ROCML2005/papers/fieldsend2CRC.pdf)

[5] Average Precision @K [link](https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf)

