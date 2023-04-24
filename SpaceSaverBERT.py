"""
This module, SpaceSaverBERT, is intended to save on space needed to store BERT deep learning models. In its current iteration, SpaceSaverBERT
is written to work with any Hugging Face NLP model.
"""

# import needed packages
import torch
import numpy as np
import dask
import dask.array as da
import time


# functions
def save_space(keeper_list, reduce_list, size):
    """
    The purpose of this function is to reduce a list to labels referencing another list so that it can be stored as a smaller list of labels.
    
    Inputs:
    keeper_list   the full-length dask array to be kept in storage
    reduce_list   the full-length dask array to be reduced to a smaller list of labels
    size          the size of chunk to use
    
    Outputs:
    keeper_chunklist  the keeper list in a 2d list form, by chunk
    reduce_labels     the short list of chunk labels for reduce_list, where each element is a number referencing a chunk in keeper_list
    """
    
    st = time.time()
    
    # split keeper list into chunks of the specified size
    keeper_chunklist = list(np.zeros(int(len(keeper_list)/size)))

    for i in range(len(keeper_chunklist)):
        keeper_chunklist[i] = np.array(keeper_list[i*size:(i*size)+size]) # making this an array makes code far more efficient
    
    # report time elapsed
    elapsed_time = time.time() - st
    print("chunked keeper; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # split reduce list into chunks of the specified size
    reduce_chunklist = list(np.zeros(int(len(reduce_list)/size)))

    for i in range(len(reduce_chunklist)):
        reduce_chunklist[i] = np.array(reduce_list[i*size:(i*size)+size]) # making this an array makes code far more efficient
    
    # report time elapsed
    elapsed_time = time.time() - st
    print("chunked reducer; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        
    # using a loss function (MSE) find which of the keeper list chunks best matches each chunk
    # of the reduce list and return the label of that chunk in a list

    min_loss = 10000
    min_label = 0
    reduce_labels = list(np.zeros(len(reduce_chunklist)))

    for i in range(len(reduce_chunklist)): # for each reduce chunk
        for j in range(len(keeper_chunklist)): # run through each keeper chunk
            mse = np.square(np.subtract(reduce_chunklist[i], keeper_chunklist[j])).mean() # calculate mse for each keeper chunk, for this reduce chunk
            if min_loss > mse: # if the new mse is less than the current min loss
                min_label = j # set the label to the current keeper chunk label
                min_loss = mse # and set the mse as the minimum loss
        min_loss = 10000
        if (i % 100 == 0):
            elapsed_time = time.time() - st
            print("reducing list, iteration " + str(i) + "; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        reduce_labels[i] = min_label # after running through each keeper chunk for this test chunk, take the final label as the chunk's generative label
        
    return reduce_labels # no longer needs to return the keeper chunklist - will not need to be stored! Regenerated in next function


def generate_list(keeper_layer, reduce_labels, size):
    """
    The purpose of this function is to generate a list from labels referencing another list.
    
    Inputs:
    keeper_chunklist   the keeper layer in its raw tensor form
    reduce_labels      the short list of chunk labels for a reduced list, where each element is a number referencing a chunk in keeper_list
    size               the size of chunks
    
    Outputs:
    reduce_list_generated    the generated list from chunk labels that resembles the original reduced list
    """
    
    st = time.time()
    
    # convert keeper layer into a one-dimensional dask array
    keeper_layer_da = da.from_array(keeper_layer.detach().cpu().numpy()).flatten()
    
    # split keeper layer into chunks of the specified size
    keeper_chunklist = list(np.zeros(int(len(keeper_layer_da)/size)))

    for i in range(len(keeper_chunklist)):
        keeper_chunklist[i] = np.array(keeper_layer_da[i*size:(i*size)+size]) # making this an array makes code far more efficient
    
    # report time elapsed
    elapsed_time = time.time() - st
    print("chunked keeper; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # loop through label list and return the chunks of the keeper list
    #reduce_labels_da = da.from_array(reduce_labels.detach().cpu().numpy()).flatten()
    generated = list(np.zeros(len(reduce_labels)))

    for i in range(len(reduce_labels)):
        generated[i] = keeper_chunklist[reduce_labels[i]]
        
    # report time elapsed
    elapsed_time = time.time() - st
    print("layer generated; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        
    # flatten the generated list
    reduce_list_generated = [j for sub in generated for j in sub]
    
    # report time elapsed
    elapsed_time = time.time() - st
    print("layer flattened; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    return reduce_list_generated



def list_to_tensor(weights_list, tensor_dims):
    """
    The purpose of this function is to generate a PyTorch tensor from a list of layer parameters.
    
    Inputs:
    weights_list      the list of layer parameters
    tensor_dims       the dimensions of the desired tensor, as an array
    
    Outputs:
    new_layer_tensor    the generated PyTorch tensor of the specified size
    """
    
    # convert list to numpy array
    new_layer_array = np.array(weights_list)
    
    # convert numpy array to correct shape
    new_layer_array = new_layer_array.reshape(tensor_dims)
    
    # convert numpy array to PyTorch tensor
    new_layer_tensor = torch.from_numpy(new_layer_array)
    
    return new_layer_tensor



def save_space_opt(keeper_list, reduce_list, size, threshold):
    """
    The purpose of this function is to reduce a list to labels referencing another list so that it can be stored as a smaller list of labels. It only reduces parts of the list that have an appropriately small MSE.
    
    Inputs:
    keeper_list   the full-length dask array to be kept in storage
    reduce_list   the full-length dask array to be reduced to a smaller list of labels
    size          the size of chunk to use
    
    Outputs:
    keeper_chunklist  the keeper list in a 2d list form, by chunk
    reduce_labels     the short list of chunk labels for reduce_list, where each element is a number referencing a chunk in keeper_list
    """
    
    st = time.time()
    
    # split keeper list into chunks of the specified size
    keeper_chunklist = list(np.zeros(int(len(keeper_list)/size)))

    # make keeper list into 2D array, where index refers to each chunk
    for i in range(len(keeper_chunklist)):
        keeper_chunklist[i] = np.array(keeper_list[i*size:(i*size)+size]) # making this an array makes code far more efficient
    
    # report time elapsed
    elapsed_time = time.time() - st
    print("chunked keeper; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # split reduce list into chunks of the specified size
    reduce_chunklist = list(np.zeros(int(len(reduce_list)/size)))

    # make reducer list into 2D array, where index refers to each chunk
    for i in range(len(reduce_chunklist)):
        reduce_chunklist[i] = np.array(reduce_list[i*size:(i*size)+size]) # making this an array makes code far more efficient
    
    # report time elapsed
    elapsed_time = time.time() - st
    print("chunked reducer; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        
    # using a loss function (MSE) find which of the keeper list chunks best matches each chunk
    # of the reduce list and return the label of that chunk in a list

    min_loss = 10000
    min_label = 0
    reduce_labels = list(np.zeros(len(reduce_chunklist)))

    for i in range(len(reduce_chunklist)): # for each reduce chunk
        for j in range(len(keeper_chunklist)): # run through each keeper chunk
            mse = np.square(np.subtract(reduce_chunklist[i], keeper_chunklist[j])).mean() # calculate mse for each keeper chunk, for this reduce chunk
            if min_loss > mse: # if the new mse is less than the current min loss
                min_label = j # set the label to the current keeper chunk label
                min_loss = mse # and set the mse as the minimum loss
        min_loss = 10000
        if (i % 100 == 0):
            elapsed_time = time.time() - st
            print("reducing list, iteration " + str(i) + "; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        final_mse = np.square(np.subtract(reduce_chunklist[i], keeper_chunklist[min_label])).mean() # get final (min) mse
        if final_mse <= threshold: # if the final mse is lower than or equal to the threshold
            reduce_labels[i] = min_label # take the final label as the chunk's generative label
        else:
            reduce_labels[i] = reduce_chunklist[i]
        
    return reduce_labels # no longer needs to return the keeper chunklist - will not need to be stored! Regenerated in next function



def generate_list_opt(keeper_layer, reduce_labels, size):
    """
    The purpose of this function is to generate a list from labels referencing another list after using space_saver_opt.
    
    Inputs:
    keeper_chunklist   the keeper layer in its raw tensor form
    reduce_labels      the short list of chunk labels for a reduced list, where each element is a number referencing a chunk in keeper_list
    size               the size of chunks
    
    Outputs:
    reduce_list_generated    the generated list from chunk labels that resembles the original reduced list
    """
    
    st = time.time()
    
    # convert keeper layer into a one-dimensional dask array
    keeper_layer_da = da.from_array(keeper_layer.detach().cpu().numpy()).flatten()
    
    # split keeper layer into chunks of the specified size
    keeper_chunklist = list(np.zeros(int(len(keeper_layer_da)/size)))

    for i in range(len(keeper_chunklist)):
        keeper_chunklist[i] = np.array(keeper_layer_da[i*size:(i*size)+size]) # making this an array makes code far more efficient
    
    # report time elapsed
    elapsed_time = time.time() - st
    print("chunked keeper; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # loop through label list and return the chunks of the keeper list
    #reduce_labels_da = da.from_array(reduce_labels.detach().cpu().numpy()).flatten()
    generated = list(np.zeros(len(reduce_labels)))

    for i in range(len(reduce_labels)):
        # if the index points to a label, index into the saved list; otherwise, keep the array at that index
        if isinstance(reduce_labels[i], int):
            generated[i] = keeper_chunklist[reduce_labels[i]]
        else:
            generated[i] = reduce_labels[i]
    
    # report time elapsed
    elapsed_time = time.time() - st
    print("layer generated; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        
    # flatten the generated list
    reduce_list_generated = [j for sub in generated for j in sub]
    
    # report time elapsed
    elapsed_time = time.time() - st
    print("layer flattened; execution time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    return reduce_list_generated



def get_avg_mse(layer_a, layer_b, size, fn_range):
    """
    The purpose of this function is to produce an average MSE value for the first chunk of the keeper layer with the first several chunks of the reducer list, exact number of chunks specified by fn_range.
    
    Inputs:
    layer_a      the keeper layer in dask array form
    layer_b      the reducer layer in dask array form
    size         the size of chunks we are considering using
    fn_range     the number of chunks to be considered in creating the average
    
    Outputs:
    avg_mse      the calculated average MSE value
    """
    mse_list = []
    for i in range(fn_range):
        mse_list.append(np.square(np.subtract(np.array(layer_a[0:size]), np.array(layer_b[i*size:size + size*i]))).mean())
    avg_mse = round(sum(mse_list)/len(mse_list), 10)
    return avg_mse
