# Automatic Question Generator
A tool to generate questions automatically from the text.

## Installation
Using pip to install the dependencies and packages
```
pip3 install -r requirements.txt
```
        
## Data Preprocessing
``` 
python3 dataset_preparation.py
```
This will generate a processed data with a filename given in the code to train our model.

## Train L-LDA model and get questions
```
python3 main.py
```
This will train the L-LDA using the dataset generated in the previos section. Make sure pre-processed dataset text file has been generated before running this command.

After all this a csv file with name best_questions will be generated having top 15 questions.