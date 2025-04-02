# DT-DHG

demo
```bash
python train.py --config configs/test/SAGE.yaml --dataset EN --name test --n_layers 5
```

@ARTICLE{10632586,
  author={Xu, Peng and Wei, Zhitao and Li, Chuchu and Yuan, Jiaqi and Liu, Zaiyi and Liu, Wenbin},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Drug-Target Prediction Based on Dynamic Heterogeneous Graph Convolutional Network}, 
  year={2024},
  volume={28},
  number={11},
  pages={6997-7005},
  abstract={Novel drug-target interaction (DTI) prediction is crucial in drug discovery and repositioning. Recently, graph neural network (GNN) has shown promising results in identifying DTI by using thresholds to construct heterogeneous graphs. However, an empirically selected threshold can lead to loss of valuable information, especially in sparse networks, a common scenario in DTI prediction. To make full use of insufficient information, we propose a DTI prediction model based on Dynamic Heterogeneous Graph (DT-DHG). And progressive learning is introduced to adjust the receptive fields of node. The experimental results show that our method significantly improves the performance of the original GNNs and is robust against the choices of backbones. Meanwhile, DT-DHG outperforms the state-of-the-art methods and effectively predicts novel DTIs.},
  keywords={Drugs;Diffusion tensor imaging;Bioinformatics;Heterogeneous networks;Graph convolutional networks;Enzymes;Training;Drug-target interaction;dynamic heterogeneous graph;graph convolutional network;link prediction;progressive learning},
  doi={10.1109/JBHI.2024.3441324},
  ISSN={2168-2208},
  month={Nov},}
