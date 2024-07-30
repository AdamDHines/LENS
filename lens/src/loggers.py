import os
import torch
import logging

from datetime import datetime

def model_logger(model): 
    """
    Configure the model logger
    """   
    now = datetime.now()
    output_base_folder = './lens/output/'
    model.output_folder = output_base_folder + now.strftime("%d%m%y-%H-%M-%S")

    # Create the base output folder if it does not exist
    os.makedirs(output_base_folder, exist_ok=True)
    
    # Create the specific output folder
    os.mkdir(model.output_folder)
    # Create the logger
    model.logger = logging.getLogger("LENS")
    if (model.logger.hasHandlers()):
        model.logger.handlers.clear()
    # Set the logger level
    model.logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename=os.path.join(model.output_folder, 'lens.log'),
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    # Add the logger to the console (if specified)
    model.logger.addHandler(logging.StreamHandler())
    model.logger.info('')
    model.logger.info('██╗░░░░░███████╗███╗░░██╗░██████╗')
    model.logger.info('██║░░░░░██╔════╝████╗░██║██╔════╝')
    model.logger.info('██║░░░░░█████╗░░██╔██╗██║╚█████╗░')
    model.logger.info('██║░░░░░██╔══╝░░██║╚████║░╚═══██╗')
    model.logger.info('███████╗███████╗██║░╚███║██████╔╝')
    model.logger.info('╚══════╝╚══════╝╚═╝░░╚══╝╚═════╝░')  
    model.logger.info('')                                         
    model.logger.info('\\\\\\\\\\\\\\\\\\\\\\\\')
    model.logger.info('LENS: Locational Encoding with Neuromorphic Systems v0.1.0')
    model.logger.info('Queensland University of Technology, Centre for Robotics')
    model.logger.info('')
    model.logger.info('© 2024 Adam D Hines, Michael Milford, Tobias Fischer')
    model.logger.info('MIT license - https://github.com/AdamDHines/LENS')
    model.logger.info('\\\\\\\\\\\\\\\\\\\\\\\\')
    model.logger.info('')
    if not model.event_driven and not model.simulated_speck:
        model.logger.info(f'Current device is CPU')
        device = torch.device("cpu")
    elif model.train_model:
        if torch.cuda.is_available():
            model.logger.info(f'Current device is {torch.cuda.get_device_name(torch.cuda.current_device())}')
            device = torch.device("cuda")
        else:
            model.logger.info(f'Current device is CPU')
            device = torch.device("cpu")
    else:
        model.logger.info('Current device is: Speck2fDevKit')
        device = torch.device("cpu")
    model.logger.info('')

    return device