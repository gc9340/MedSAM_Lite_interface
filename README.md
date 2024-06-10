# MedSAM_Lite_interface


A easy to use web browser interface for MedSAM Lite. 


## Installation

This was developed and tested mainly on Windows 11 and python 3.10.14. Some testing was also
done on Ubuntu 24.04.


Step 0. If you haven't already, follow the instructions for MedSAM Lite to get a trained `medsam_lite_latest.pth` file.

1. Clone the repository, `git clone -b MedSAM_Interface https://github.com/gc9340/MedSAM_Lite_interface/`
2. Go into the MedSAM_Lite_interface folder
3. Create a virtual environment using conda `conda env create --name medsam_interface -f interface.yml`
4. Activate the environment `conda activate  medsam_interface`
5. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
6. Place `medsam_lite_latest.pth` directly inside the `UHN` folder
7. Run `python manage.py runserver`
8. Open a web browser and navigate to http://localhost:8000

## Usage

Hopefully you should see something like this

![Alt text](https://github.com/gc9340/MedSAM_Lite_interface/blob/main/home_page.png?raw=true)

This interface works on the same data that the MedSAM Lite setup tutorial uses. You can find some data examples in `interface/npz_examples/`. 
Since that was involving 3D inference, the inference for 2D has not been thoroughly tested, so be wary of errors and bugs when running 2D inference. 

1. Click on the choose file button on the top left, and select an .npz file.
2. Select the 3D inference option on the radio button
3. Select a device (cuda is prefered)  
4. Click the `Run MedSam On File` button on the bottom left
5. Wait for the server to finish performing inference and saving a .png (should be a few seconds or so)
6. The entry should now appear in the inference history table on the right

![Alt text](https://github.com/gc9340/MedSAM_Lite_interface/blob/main/upload.png?raw=true)
7. Click on the file name in the table to open up the entry in a modal

8. Use the modal to view the png or delete the entry

![Alt text](https://github.com/gc9340/MedSAM_Lite_interface/blob/main/modal.png?raw=true)



