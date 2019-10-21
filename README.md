# Duplicate Identifcation in Geographic Data

This project is intended to sort through the Oklahoma Landmarks Inventory Database and identify 
duplicate/similar properites cemetaries. I intend to do this through the use of training a 
Supervised Binary Nerual Network Classifier. Libaries that will be used for this project are
Tensorflow to perform the neural network calculations, pandas to observe the data, and matplotlib
to visuialize.

# Objectives 
<ol>
	<li>Explore data</li>
	<li>Preproccess</li>
	<li>Short list model assumptions</li>
	<li>Construct framework</li>
	<li>Test iterations</li>
	<li>Evalute and retest</li>
	<li>Present Results</li>
</ol>

# The Dataset:

This data is from the 1970s and has been converted between storage mediums several times.
From index cards to floppy dics, onto a relational database. Records have been copied by hand,
from Object Character Recognizion and retyped in by hand. 

There are 67 fields of a varity of catgorical data, interval ratio data, and numerous null values.

One of the fields is marked as poss_dup or good. This catagoriy duplicate_check is what will
be used to train the machine learning model.

# End Product

A CSV file with the ObjectID (primary key) for each record and the probablity of a duplicate 
record given.

# Value

This will enable the dataset owners to make better decisions on what records to remove based
on what expert classifers have done.

# Reaserch Proccess

#### Put this into the README.md file:
## Objectives:
<ol>
    <li>Name attributes and describe characteristics.</li>
    <ul>
        <li>% of Nulls</li>
        <li>Type of data. ie String, int, float</li>
        <li>Noise present. Such as outliers, logistic, rounding errors</li>
        <li>What is useful what isn't and why</li>
        <li>Type of distribution. </li>
    </ul>
    <li>Identifying Label data</li>
    <li>Visualization of data</li>
    <li>Identify correlations between variables</li>
    <li>Propose how the problem would be solved manually</li>
    <li>Provide transformations if nessiary</li>
    <li>Anything else of interest</li>
</ol>

## Exploritory data Strategy
The nature of the dataset is complex. This is due to the descriptive attirbutes assocatied with properites and cemetaries. There are only a couple of real numerical datatypes such as lat and long. I intend to go through each type figuring out if it is catagorical, a string/text, or numerical. Once the catagroies are identified applying a numerical number scheme for them will be adopted. 

### Questions:
Can I even descrptive statstics on catagorical data? From Comer's class I remember there beng some very strang things that happened.
What do I even do with the catagorical data?

# Models
<ol>
    <li>Turn everything into text and concatenate all attributes into one string, apply TF-IDF and cosine similarity, then run model.</li>
    <ul>
        <li><b>Notes</b></li>
        <ul>
            <li>I do not know how and if I need to do describptive stats on the vectors of the strings</li>
        </ul>
    </ul>
    <li>Exploritiory driven modeling</li>
    <ul>
        <li>Convert individual columns into vectors</li>
        <li>PCA (principle compoent analysis vectors and drop relivant columns</li>
        <li>Train model</li>
    </ul>
    <li></li>
