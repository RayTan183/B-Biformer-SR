# Reproduction of Resshift

# Download the project

 - The project of Resshift is [here](https://github.com/zsyOAOA/ResShift)

# Environment

 - Python 3.10, Pytorch 2.1.2, xformers 0.0.23 
 - More detail (See environment.yml) A suitable conda environment named resshift can be
   created and activated with:
   
```python
> conda create -n resshift python=3.10
conda activate resshift
pip install -r requirements.txt
```

# Download the weight
Download the model weight resshift_bicsrx4_s4.pth [here](https://github.com/zsyOAOA/ResShift/releases/tag/v2.0)

# Generate the SR images
```python
> python inference_resshift.py -i [image folder/image path] -o [result folder] --task bicsr --scale 4
```
