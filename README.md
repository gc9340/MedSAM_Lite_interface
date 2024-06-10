# MedSAM_Lite_interface


A easy to use web browser interface for MedSAM Lite. 


## Installation

This was developed and tested mainly on Windows 11 and python 3.10.14. Some testing was also
done on Ubuntu 24.04.


1. Create a virtual environment `conda create -n medsam python=3.10.14 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. Create and navigate to an empty directory called `interface`
4. While in `interface`, `git clone -b MedSAM_Interface https://github.com/gc9340/MedSAM_Lite_interface/`
5. Run `pip install -e .`
6. `python manage.py makemigrations`
7. `python manage.py migrate`
8. `python manage.py runserver`
9. Open a web browser and navigate to http://localhost:8000

