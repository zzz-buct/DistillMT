# Supplementary Materials for "A Knowledge Distillation-Based Approach for Line-Level Defect Prediction"

<p align="center">
  <img src="Figures/The framework of DistillMT.png" width="800">
</p>

## Prerequisites

Install the necessary dependencies before running the project:
<a name="ZRVYs"></a>
### Environment Requirements
Here are the suggested environment:
```
- python==3.9.13
- pytorch==1.12.1
- dgl==1.0.2+cu116
- ogb==1.3.5
- sklearn
- torch-cluster==1.6.0
- torch-scatter==2.0.9
- torch-sparse==0.6.15
- torch-geometric==2.1.0
- torchvision==0.13.1
```
<a name="G2LdD"></a>

### Thrid Party Liraries



- [PropertyGraph](https://github.com/Zanbrachrissik/PropertyGraph)

Additionally, replace `graph_tool_path` in `script/1.1 HPDG Construction_PDG_Code.py` with your own PropertyGraph path.

## Datasets

The datasets are obtained from Wattanakriengkrai et. al. The datasets contain 32 software releases across 9 software projects. The datasets that we used in our experiment can be found in this [github](https://github.com/awsm-research/line-level-defect-prediction).

The file-level datasets (in the File-level directory) contain the following columns

 - `File`: A file name of source code
 - `Bug`: A label indicating whether source code is clean or defective
 - `SRC`: A content in source code file

The line-level datasets (in the Line-level directory) contain the following columns
 - `File`: A file name of source code
 - `Commit`: A commit id of bug-fixing commit of the file
 - `Line_number`: A line number where source code is modified
 - `SRC`: An actual source code that is modified

For each software project, we use the oldest release to train DistillMT models. The subsequent release is used as validation sets. The other releases are used as test sets.

For example, there are 5 releases in ActiveMQ (e.g., R1, R2, R3, R4, R5), R1 is used as training set, R2 is used as validation set, and R3 - R5 are used as test sets.

## Run the code
1. Construct the HPDG (Execution of scripts in the order of 1.0-1.5):
```
    python 1.* HPDG Construction_*.py
```
2. Run the Teacher Model(Global Semantic-Aware Teacher Model Training):
```
    bash train_all_teachers.sh
```
3. Run the Student Model(Knowledge Distillation-Based Student Model Training):
```
    bash train_all_students.sh
```
4. Get The Multi-granularity Prediction:
```
    bash test_all_students.sh
```
