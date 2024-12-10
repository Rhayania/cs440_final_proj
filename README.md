# Genomic Variant Classification with Neural Networks
## Author: Kira Vasquez-Kapit

### Summary
This project is beginning as a final course assignment in CS 440: Intro to AI at Colorado State University. However, I intend to continue work on this after the semester is over. It uses a recurrent neural network (RNN) built to classify variation between two DNA sequences. The following classifications are considered:
- No variation
- Indel (an insertion or deletion)
- Inversion
- Single nucleotide polymorphism (SNP, or single-base difference)

### Directory Navigation
Data: Normally, I discourage the uploading of data to GitHub. However, I thought it was important for my training and testing data to be included so that the model can be run without anyone needing to generate new data.
 - basic_training: 40 .txt files of training data, with filenames describing the type of variation within. Each .txt file has two lines of sequences to be compared.
 - basic_validation: 40 .txt files of validation data, set up as above.
 - basic_testing: 40 .txt files of testing data, set up as above.

Scripts: Directory where Python scripts are kept. For now, there is only my original neural network model.

### Dependencies
This project requires the folllowing software:
- Python 3
- Numpy
- PyTorch

### Running the Project
As long as you have downloaded the GitHub repository and the above dependencies are installed in your current environment, you may run the project **from within the _scripts_ directory** with:
`python nn_model.py`

### Known Issues
The model is currently attemping to classify each base in a sequence, instead of the entire sequence itself. I have not yet found a workaround or solution for this, and I will need to consult more experienced ML/AI experts in the coming weeks.

### Future Work

Assuming the known issues are resolved and the model's classification accuracy is relatively high, I will move on to longer sequnces with complex and numerous instances of variation. Stay tuned for more!
