# Udacity Second Project - Disaster Response Pipeline
My repository is dedicated to my second Udacity Nanogram Data Scientist project where I clean disaster response data, create a ML model to classify them and visual data in a local web. There will also be a function in the web where you can insert a response text and have it classified.

## Table of contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Description](#file)
4. [Instruction](#instruction)
5. [Results](#results)
6. [Acknowledgements](#acknowledgements)

### Instalation <a name="installation"></a>
Run the following command in your directory to have packages installed:  
`pip install -r requirements.txt`

### Project Motivation <a name="motivation"></a>
The purpose of this project is to create a website that:
- Display visuals about a disaster messages collected from disaster response dataset. Users will understand more about the messages that were sent out.
- Allow users to insert any message and return the categories that this message will be classified into. Users can utilize this to re-direct the requesters to the correct aiding agencies.  

### File Description <a name="file"></a>
Data folder includes 3 files:
- disaster_messages.csv contains message dataset
- disaster_categories.csv contains categories of the messages
- process_data.py contain the ETL pipeline code to clean, transform and preprocess the above files into final database

Models folder include 1 file:
- train_classifier.py contain the ML pipeline code to apply NLP & ML to train, fit and evaluate the model, then return the best model into a pickel file

App folder include 3 files:
- templates/go.html & templates/master.html contain the html code for the design of the website
- run.py contains the code for the processes running behind the website

### Instruction <a name="instruction"></a>
Run the following commands in the project's root directory to set up your database and model.
- Step 1: Run ETL pipeline that cleans data and stores in database. Output database file will be saved at data/DisasterResponse.db  
    If you choose to save your database in a different name/directory, you need to change the database file path in your command.
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- Step 2: Run ML pipeline that trains classifier and saves. Output pickel file will be saved at models/classifier1.pkl.  
    If you choose to save your database/model in a different name/directory, you need to change the database.model file path in your command.
        `python models/train_classifier.py data/DisasterResponse.db models/classifier1.pkl`
- Step 3: Run the app.  
    If you choose to save your database or model in a different name/directory, you need to change the database file path or model filepath in run.py file.  
        `python app/run.py`
- Step 4: Access the link provided in your command line window to see and use the web!

### Results <a name="results"></a>
You can see in the 1st chart that only 'direct' category have non-English messages and there's no non-English messages in 'social' or 'news'. This indicates that the data collected maybe not representative enough.  
In the 2nd chart we can see that there are many types with very little messages. This indicates that the data is imbalanced and the model may work well with this dataset but will not work as good in real situation.

We should either try to collect better data or use some other techniques (resampling, penalized model,..) to resolve this issue.

### Acknowledgements <a name="acknowledgements"></a>
The code is inspired by Udacity's Data Scientist Nanodegree Program.  
Data was provided by Figure Eight.
