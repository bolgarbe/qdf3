# QDF3

QDF3 is a multiple kernel learning-based support vector machine (SVM) and distance metric learning (DML) solver for drug prioritization and repositioning.

## Usage

```
Usage: 
   -c MKL parameter C list (e.g. 1,10,100,1000, default: 100)
   -e MKL parameter epsilon (default: 0.0001)
   -l MKL parameter lambda (default: 1)
   -f Output file prefix (default: test)
   -i cv iterations (default: 1, don't do cv)
   -k kernel file
   -t training file (queries)
   -r do probability estimates
Kernel file format:
   path_to_kernel_1 type_1 param_11,param12,param13,...
   path_to_kernel_2 type_2 param_21,param22,param23,...
   ...
Types:
   0=cosine, 1=Tanimoto, 2=RBF, 3=linear, 4=quadratic, 5=polynomial
Training file format:
   +q1entity1,+q1entity2,...;-q1entity1,-q1entity2,...
   +q2entity1,+q2entity2,...;-q2entity1,-q2entity2,...

Example: qdf -k mykernels.txt -t myqueries.txt -f pr1
```

## Publications

B Bolgár, P Antal
Towards Multipurpose Drug Repositioning: Fusion of Multiple Kernels and Partial Equivalence Relations Using GPU-accelerated Metric Learning.
In: Jobbágy, Ákos (szerk.) First European Biomedical Engineering Conference for Young Investigators : ENCY 2015 Singapore : Springer (2015) pp. 36-39. , 4 p.

Arany, A ; Bolgar, B ; Balogh, B ; Antal, P ✉ ; Matyus, P
Multi-aspect candidates for repositioning: data fusion methods using heterogeneous information sources.
CURRENT MEDICINAL CHEMISTRY 20 : 1 pp. 95-107. , 13 p. (2013)

## License
[MIT](https://choosealicense.com/licenses/mit/)