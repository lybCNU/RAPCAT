# RAPCAT
The Robust-point-matching- And Piecewise-affine-based Cell Annotation Tool (RAPCAT) for annotation of all 558 cells in L1-stage Caenorhabditis elegans image stack

# Usage
Once downloaded, go to `RAPCAT-main` folder with matlab, and run `RAPCAT('exampleData/')` in matlab command prompt for annotation of the example image stack, i.e., `nhr-206_XIL0604_20140708_0001_S1`.

After cell annotation is finished, you can visualize the image stack and annotation result using software VANO  in `vano_win32_1.741` folder. At first, open the VANO by promting `wano.exe`. Next, choose layout of stacking as two rows by pressing `No` button. Then
drag the `nhr-206_XIL0604_20140708_0001_S1.ano` or `nhr-206_XIL0604_20140708_0001_S1_recog.ano` file to VANO interface.

The `traindata` folder contains all data for atlas generation, RAPCAT training and atlas for RAPCAT annotation. More dataset can be found at `https://doi.org/10.5281/zenodo.7627915`.

# Questions & troubleshooting
If you have some problems or find some bugs in the codes, please email: li.yongbin@cnu.edu.cn
