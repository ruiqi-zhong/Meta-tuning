import os

def get_id_from_path_name(p: str):
    bname = os.path.basename(p)
    return int(bname.split('_')[0].replace('group', ''))


def get_name_from_path_name(p):
    bname = os.path.basename(p)
    return bname.split('_')[1].split('.')[0]

training_dicts_dir = 'data/training_dicts/'
testing_dicts_dir = 'data/testing_dicts/'