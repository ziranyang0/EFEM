# conda remove -n Efem --all -y
# conda create -n Efem python=3.10 -y
# source activate Efem
conda activate Efem

# install pytorch
echo ====INSTALLING PyTorch======
which python
which pip
# conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# # install pytorch3d
echo ====INSTALLING=PYTORCH3D======
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
# conda install pytorch3d -c pytorch3d -y
conda install pytorch3d=0.6.1 -c pytorch3d -y

# # Install Pytorch Geometry
conda install pyg -c pyg -y

# install requirements
pip install cython
pip install -r requirements.txt
# pip install pyopengl==3.1.5 # for some bugs in the nvidia driver on our cluster

# build ONet Tools
cd lib_shape_prior
python setup.py build_ext --inplace
python setup_c.py build_ext --inplace # for kdtree in cuda 11