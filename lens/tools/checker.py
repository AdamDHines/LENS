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

    def calculate_best_conv_params(input_height, input_width, target_output_height, target_output_width):
        """
        Calculate convolution parameters that achieve the closest possible output size
        to the target, using only Conv2d with stride=kernel_size.
        
        Returns parameters and the actual achievable output size.
        """
        
        def find_best_output_dim(input_dim, target_output_dim):
            """Find the best achievable output dimension for a single axis."""
            # For Conv2d: output = floor((input - kernel) / stride) + 1
            # With stride = kernel: output = floor((input - kernel) / kernel) + 1
            #                              = floor(input/kernel - 1) + 1
            #                              = floor(input/kernel)
            
            # So kernel = input / output (approximately)
            
            # Find candidate kernel sizes around the ideal
            ideal_kernel = input_dim / target_output_dim
            
            # Test kernel sizes around the ideal
            candidates = []
            for kernel in range(max(1, int(ideal_kernel) - 2), int(ideal_kernel) + 3):
                if kernel > 0 and kernel <= input_dim:
                    # Calculate actual output with this kernel
                    output = (input_dim - kernel) // kernel + 1
                    if output > 0:
                        # Calculate how close this is to target
                        diff = abs(output - target_output_dim)
                        candidates.append({
                            'kernel': kernel,
                            'output': output,
                            'diff': diff,
                            'ratio_error': abs(output / target_output_dim - 1.0)
                        })
            
            # Sort by difference from target, then by ratio error
            candidates.sort(key=lambda x: (x['diff'], x['ratio_error']))
            
            return candidates[0] if candidates else {'kernel': 1, 'output': input_dim, 'diff': float('inf')}
        
        # Find best parameters for each dimension
        best_h = find_best_output_dim(input_height, target_output_height)
        best_w = find_best_output_dim(input_width, target_output_width)
        
        # Calculate total neurons
        actual_output_h = best_h['output']
        actual_output_w = best_w['output']
        target_neurons = target_output_height * target_output_width
        actual_neurons = actual_output_h * actual_output_w
        
        return {
            'kernel_size': (best_h['kernel'], best_w['kernel']),
            'stride': (best_h['kernel'], best_w['kernel']),  # stride = kernel for non-overlapping
            'padding': 0,
            'target_output_size': (target_output_height, target_output_width),
            'actual_output_size': (actual_output_h, actual_output_w),
            'target_neurons': target_neurons,
            'input_neurons': actual_neurons,
            'neuron_difference': actual_neurons - target_neurons,
            'dimension_errors': {
                'height': best_h['diff'],
                'width': best_w['diff']
            }
        }
    
    return calculate_best_conv_params(args.roi_dim[0], args.roi_dim[1], args.dims[0], args.dims[1])