import sys
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to read in messages and categories data from .csv files 
    and then combine them into df dataframe.

    Args:
        messages_filepath (string): location of messages dataset
        categories_filepath (string): location of categories dataset

    Returns:
        df: merged dataframe

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories)
    return df



def clean_data(df):
    """ Function to clean a dataframe by converting categories into numeric columns and remove duplications 
    
    Args:
        df (object): original dataframe

    Returns:
        df: cleaned dataframe
    """
    # create a dataframe of the individual category columns
    categories = df['categories'].str.split(';',expand=True) 
    
    # use the categories in the first column to create new column names
    row = categories.iloc[0]
    category_colnames = list(map(lambda category: str(category)[0:-2], row))
    categories.columns = category_colnames
    
    # convert the category values to numeric
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
        
    # replace the categories column in df with the new category columns
    df.drop('categories', inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates
    df.drop_duplicates(keep=False,inplace=True)
    return df


def save_data(df, database_filename):
    """Function to save a dataset df into an sqlite database.

    Args:
        df (object): a dataframe you want to save
        database_filename (string): the name of sqlite database output

    Returns:
        None

    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename, engine, index=False)
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()