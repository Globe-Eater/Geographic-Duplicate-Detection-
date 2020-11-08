# Duplicate Identifcation in Geographic Data

## Setup:
<ol>
    <li>Create a new directory (folder) somewhere convenient.</li>
    <li>Download the zip file, move to the convenient space and unzip.</li>
    <li>If anaconda has not been downloaded do so now. If it has run the following commands
    in the anaconda command prompt:</li>

      conda create env -f setup/enviorment.yml
      conda activate SHPO 
      conda list to see if you have tensorflow version 1.13
</ol>

## Usage:
<p> To trained a new model use: </p>
    python All_in_One.py
<p> This will create a new model and save it to the file that All_in_One.py is in. </p>
    
## Expectations:
<p> This model is capable of classifying duplicate records at a rate of 89.5% of the time. </p>
    
    
