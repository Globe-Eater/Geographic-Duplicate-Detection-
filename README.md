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
