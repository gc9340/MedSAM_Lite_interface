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

## Usage

Hopefully you should see something like this



This interface works on the same data that the MedSAM Lite setup tutorial uses. You can find some data examples in `interface/npz_examples/`. 
Since that was involving 3D inference, the inference for 2D has not been thoroughly tested, so be wary of errors and bugs when running 2D inference. 

1. Click on the choose file button on the top left, and select an .npz file.
2. Select the 3D inference option on the radio button
3. Select a device (cuda is prefered)  
4. Click the `Run MedSam On File` button on the bottom left
5. Wait for the server to finish performing inference and saving a .png (should be a few seconds or so)
6. The entry should now appear in the inference history table on the right
7. Click on the file name in the table to open up the entry in a modal
8. Use the modal to view the png or delete the entry





