
import os
import torch

def create_unique_directory(directory_name):
    # Check if the original directory already exists
    if not os.path.exists(directory_name):
        # If not, create the original directory
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created.")
        new_directory_name = directory_name
    else:
        # If the original directory exists, find a unique name
        count = 1
        while True:
            new_directory_name = f"{directory_name}_{count}"
            # Check if the new directory name exists
            if not os.path.exists(new_directory_name):
                # If not, create the new directory
                os.makedirs(new_directory_name)
                print(f"Directory '{new_directory_name}' created.")
                break
            else:
                # If the new directory name exists, try the next count
                count += 1
    return new_directory_name



def save_model(model, model_name) : 
    models_dir = os.getcwd() + '/models/' 
    if not os.path.exists(models_dir):
        # If not, create the original directory
        os.makedirs(models_dir)
    models_path = models_dir + model_name
    torch.save(model, models_path )

