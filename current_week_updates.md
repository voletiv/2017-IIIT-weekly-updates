---

## 2017-10-23

### DONE

- Use ROC and Operating Point as metrics for comparison

- Made and compared assessors from Head Pose + Lipreader features, with Lipreader ROC

[comp](20171023/COMPARISON_VAL.png "Frame")

[baseline_lipreader](20171023/ROC_baseline_lipreader.png "Frame")

- Done the same for lipreader trained on 10% Training Data

[logReg_10pc](20171023/ROC_10pc_logReg_unopt.png "Frame")

[baseline_lipreader_10pc](20171023/ROC_10pc_baseline_lipreader.png "Frame")

### DISCUSSION

- Better critic? Better features?

- Assessor on the Retrieval

- Assessor - Lipreader LSTM output + Head pose -> LSTM

### SUMMARY

FAILURE MODES:

- Extracted head poses on GRIDcorpus using dlib

- To compare assessors on Lipreader predictions, use:
    - ROC AUC
    - Operating point

- Using 1-dim Word Durations, 6-dim Head Poses (3 Means, 3 Ranges) , 64-dim Lipreader_Features as Attributes

- Calculated ROC for 4 different assessors:
    - C3D Critic (of old)
    - Logistic regressor
    - Linear SVM
    - RBF SVM

- Changed Operating Point from default to that closest to (0, 1) [(fpr, tpr)]

![Full comparison](20171023/COMPARISON.png "Frame")

#### TO DO:

- ROC AUC of lipreader trained on 50% and assessor on those predictions
- Self-training?

---

### ARCHIVES

[Archives](archives.html)
