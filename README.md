# Duplicate Identifcation in Geographic Data

## Setup:
<ol>
    <li>Create a new directory (folder) somewhere convenient.</li>
    <li>Download the zip file, move to the convenient space and unzip.</li>
    <li>If anaconda has not been downloaded do so now. If it has run the following commands
    in the anaconda command prompt:</li>
    ```
       conda create env -f setup/enviorment.yml
       conda activate SHPO 
       conda list to see if you have tensorflow version 1.13
    ```
</ol>

## Usage:
From the command line navigate to the directory that the convenient place is if you aren't already there.
<p> These are the steps to produce an excel file with ObjectIDs and Predictions:
<ol>
    <li>Making sure the enviroment is active by seeing (SHPO) on the left, run: </li>
    <p> `python prep.py`
    <ul>
        <li>This will require an excel file that has been checked (for duplicates).</li>
        <li>Prep will ask for a path to a location for the data to be saved place it 
        in datasets/prepared_data/County_Name.xlsx</li>
    </ul>
    <li>Next you will run</li>
    <p> `python Classifier.py`
    <ul>
        <li>This program will ask for your prepared dataset.</li>
        <li>It will then generate an xlsx file of ObjectID and its Prediction. </li>
    </ul>
</ol>

## Evaluation
    
    
