import os
import torch
import torch.nn as nn

def check_args(args):
    # Check working directories
    basepath = os.path.join(args.data_dir, args.dataset, args.camera)
    assert (os.path.exists(basepath)), "Data directory does not exist: {}".format(basepath)
    assert (os.path.exists(os.path.join(basepath,args.reference))), "Reference directory does not exist: {}".format(os.path.join(basepath,args.reference))
    if not args.train_model and not args.collect_data and not args.event_driven: # Only check query if not running inference model
        assert (os.path.exists(os.path.join(basepath,args.query))), "Query directory does not exist: {}".format(os.path.join(basepath,args.query))
    # Check that the correct number of images are in the reference and query directories
    reference_images = len(os.listdir(os.path.join(basepath,args.reference)))
    assert (args.reference_places * args.filter <= reference_images), f"Not enough reference images for {args.reference_places} places and a filter of {args.filter}"
    if not args.train_model and not args.collect_data and not args.event_driven:
        query_images = len(os.listdir(os.path.join(basepath,args.query)))
        assert (args.query_places * args.filter <= query_images), f"Not enough query images for {args.query_places} places and a filter of {args.filter}"
    # Check that a dataset does not already exist if collecting new data
    if args.collect_data:
        assert (not os.path.exists(os.path.join(basepath,args.data_name))), "Data directory already exists: {}".format(os.path.join(basepath,args.data_name))
    # Check that ROI dimensions and final image sizing is compatible
    test_tensor = torch.zeros([args.roi_dim, args.roi_dim])
    kernel_size = args.roi_dim // args.dims
    conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=kernel_size)
    output = conv(test_tensor.unsqueeze(0).unsqueeze(0))
    assert (output.shape[2] == args.dims), "ROI dimension and final image size are incompatible"