# big-data-parameter-reduction

This repository includes notebooks for reproducing our results from our parameter reduction final project in DS 5110: Big Data Systems as well as the python package we created for the project.

We do not include the data we imported in this repository due to large space needs. The models can be found on Hugging Face. To run these notebooks, they should be downloaded as state dictionaries and all paths in the notebooks should be updated accordingly.

The repository includes the following files:
- **SpaceSaverBERT.py**: the Python package we created, with methods to resize and regenerate Hugging Face models
- **SSB_Instructions.ipynb**: a walkthrough notebook demonstrating how to use the functions from SpaceSaverBERT to resize and regenerate a model
- **Accuracy_Testing.ipynb**: a walkthrough notebook demonstrating how to test an optionally-saved and force-saved version of a model with its original state dictionary
