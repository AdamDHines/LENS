import os
import torch
import logging

from datetime import datetime

def model_logger(model): 
    """
    Configure the model logger
    """   
    if os.path.isdir('../output'):
        now = datetime.now()
        model.output_folder = '../vprtemponeuro/output/' + now.strftime("%d%m%y-%H-%M-%S")
    else:
        now = datetime.now()
        model.output_folder = './vprtemponeuro/output/' + now.strftime("%d%m%y-%H-%M-%S")
    
    #os.mkdir(model.output_folder)
    # Create the logger
    model.logger = logging.getLogger("VPRTempo")
    if (model.logger.hasHandlers()):
        model.logger.handlers.clear()
    # Set the logger level
    model.logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename="/home/adam/Documents/logfile.log",
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    # Add the logger to the console (if specified)
    model.logger.addHandler(logging.StreamHandler())
        
    model.logger.info('')
    model.logger.info(' _   _____________ _____                                _   _')                      
    model.logger.info('| | | | ___ \ ___ \_   _|                              | \ | |')                     
    model.logger.info('| | | | |_/ / |_/ / | | ___ _ __ ___  _ __   ___ ______|  \| | ___ _   _ _ __ ___') 
    model.logger.info('| | | |  __/|    /  | |/ _ \  _   _ \|  _ \ / _ \______|     |/ _ \ | | |  __/ _ \\')                 
    model.logger.info('\ \_/ / |   | |\ \  | |  __/ | | | | | |_) | (_) |     | |\  |  __/ |_| | | | (_) |')
    model.logger.info(' \___/\_|   \_| \_| \_/\___|_| |_| |_| .__/ \___/      \_| \_/\___|\__,_|_|  \___/') 
    model.logger.info('                                     | |')                                           
    model.logger.info('                                     |_|')                                           

    model.logger.info('-----------------------------------------------------------------------')
    model.logger.info('VPRTempoNeuro: Neuromorphic Visual Place Recognition v0.1.0')
    model.logger.info('Queensland University of Technology, Centre for Robotics')
    model.logger.info('')
    model.logger.info('© 2023 Adam D Hines, Michael Milford, Tobias Fischer')
    model.logger.info('MIT license - https://github.com/AdamDHines/VPRTempoNeuro')
    model.logger.info('\\\\\\\\\\\\\\\\\\\\\\\\')
    model.logger.info('')
    if model.raster:
        if torch.cuda.is_available() and model.raster_device == 'gpu':
            model.logger.info(f'Current device is {torch.cuda.get_device_name(torch.cuda.current_device())}')
            device = torch.device("cuda")
        else:
            model.logger.info(f'Current device is CPU')
            device = torch.device("cpu")
    elif model.train_new_model or model.norm:
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