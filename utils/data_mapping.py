import os
import json

def create_object_mapping(master_id, object_id, image_name):
    """
    Create a mapping between object ID and image name.
    """
    return {
        "master_id": master_id,
        "object_id": object_id,
        "image_name": image_name
    }

def save_mapping(mapping, master_folder):
    """
    Save the mapping to a JSON file.
    """
    mapping_file = os.path.join(master_folder, "object_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=4)

def load_mapping(master_folder):
    """
    Load the mapping from a JSON file.
    """
    mapping_file = os.path.join(master_folder, "object_mapping.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            return json.load(f)
    return []