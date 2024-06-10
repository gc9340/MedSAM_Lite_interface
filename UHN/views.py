from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError
from UHN.models import TestFile
from django.http import JsonResponse
from UHN.medsam_interface import MedSAM_Interface
from datetime import datetime
from django.http import HttpResponseBadRequest
from cv2 import error as cv2Error
import os




PRED_IMG_PATH = 'UHN/data/pred_img/'
SEGS_PATH = 'UHN/data/segs'
MODEL_PATH = 'UHN/medsam_lite_latest.pth'


# global variables, but should really be session variables.
# As long as this is on local host it should be fine


global server_messages
server_messages = []
global interface
interface = MedSAM_Interface(MODEL_PATH)
def home(request):
    '''
    Home page for the UI.
    
    Session Variables
        - num_files (int): number of historical TestFiles to display. Default = 10 
        - order_by (str): ordering of the TestFiles (a field in the django model). Default = '-uploaded_at'
        - main_message (str): main message from the server to the front end. Default = 'No File Uploaded' 
    '''
    
    # Set session variables to defaults if they are not set
    if 'num_files' not in request.session:
        request.session['num_files'] = 10
        request.session.modified = True
    if 'order_by' not in request.session:
        request.session['order_by'] = "-uploaded_at"
        request.session.modified = True
    if 'main_message' not in request.session:
        request.session['main_message'] = 'No File Uploaded'
        request.session.modified = True
        
    # get the filtered and sorted test_files
    test_files =  TestFile.objects.all().order_by(request.session['order_by'])\
        [:request.session['num_files']]
    global server_messages
    # test_files and messages are the only things needed to update the front end
    context = {
        'test_files': test_files,
        'server_messages': server_messages,
        'main_message': request.session['main_message'],
        'order_by': request.session['order_by']
    }            
    
    # render the main html page 
    return render(request, "upload.html", context)

def validate_file_extension(file, allowed_extensions, request):
    '''
    file: a file object
    allowed_extensions: a list of allowed extensions
    
    Raise a ValidationError if file is not in allowed_extensions.
    Currently, the only extention allowed is .npz, but things may change.
    '''
    extension = file.name.split('.')[-1].lower()
    if extension not in allowed_extensions:
        raise ValidationError(f'Invalid file extension: .{extension}.\
                              Allowed extensions are: {", ".join(allowed_extensions)}')
    global server_messages
    server_messages.append('File extension extension verified')



def upload_file(request):
    '''
    Upload the npz file to server and perform inference on it.
    Create a png file based on the model inference and save it to UHN/data/pred_img/.
    Save the entry in the database as a TestFile object, and return back to home
    '''
    if request.method == 'POST': # only support post for now
        file = request.FILES.get('file-upload')
        dimension = request.POST.get("inference-type")[0]
        
        device = request.POST.get("selected-device")
        if device == "other":
            device = request.POST.get("other-device")
        allowed_extensions = ['npz']
       
        global server_messages
        server_messages = [f'File {file.name} received by server']
        try:
            validate_file_extension(file, allowed_extensions, request)
            global interface
            device_message = interface.select_device(device)
            server_messages.append(device_message)
            prediction_time = interface.get_inference(file, dimension) # run inference and get the time it takes
            server_messages.append("Inference performed successfully")
            if dimension == '3':
                interface.make_png_3D()
            else:
                interface.make_png_2D()
            server_messages.append("PNG created successfully")
            # Now that inference is done, we can save this entry to the database
            test_file = TestFile(name = file.name.split(".")[0], prediction_time = round(prediction_time, 4),\
                                 dimension = dimension, uploaded_at = datetime.now())
            test_file.save()
            
            
            server_messages.append(f'{file.name} now ready \
                                                      for viewing in inference history')
            request.session['main_message'] = 'File Processed Sucessfully'
            request.session.modified = True
        except ValidationError as e: # Bad file format
            request.session['main_message'] = 'Error Processing'
            server_messages.append("ERROR: Bad file format")
            request.session.modified = True
        
        except cv2Error as e: # Probably the wrong dimension given for the file
            request.session['main_message'] = 'Error Processing'
            server_messages.append("ERROR: Inference could not be performed. Likely cause is the wrong dimension")
            request.session.modified = True
    
    return redirect('home')

def check_exists(request, name):
    '''
    Check if a file already exists
    '''
    if request.method == "GET":
        exists = TestFile.objects.filter(pk = name).exists()
        return JsonResponse({'exists': exists})

def delete_file(request):
    '''
    Delete an inference file from the database
    '''
    if request.method == "POST":
        global server_messages
        file_name = request.POST.get("file-name")
        file = TestFile.objects.filter(pk = file_name)
        if len(file) < 1:
            request.session['main_message'] = 'Unable To Delete'
            server_messages = [f'ERROR: File {file_name} not found for deletion']
        else:
            
            if os.path.exists(f'{PRED_IMG_PATH}{file_name}.png'):
                os.remove(f'{PRED_IMG_PATH}{file_name}.png')
            if os.path.exists(f'{SEGS_PATH}{file_name}.npz'):
                os.remove(f'{SEGS_PATH}{file_name}.npz')
            file[0].delete()
            request.session['main_message'] = 'Successful Deletion'
            server_messages = [f'Deleted file {file_name}']
        request.session.modified = True
    return redirect('home')    

def sort_by(request, order_by):
    if request.method == "GET":
        print(order_by)
        request.session['order_by'] = order_by
        request.session.mofified = True
    return redirect('home')
def get_server_messages(request):
    '''
    Returns a JSON object with the server messages.
    '''
    if request.method == "GET":
        
        global server_messages
        return JsonResponse({'server_messages': server_messages})
    return HttpResponseBadRequest("Unable to get server messages")