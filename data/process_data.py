import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how = 'inner', on = 'id')
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # Extract a list of new column names for categories.
    category_colnames = pd.Series(row).apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(labels = ['categories'], axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    # a few 'related' are 2, change them to 1
    df.loc[df.related > 1, 'related'] = 1
    # There are no child alone category, dropping it
    df.drop(labels = ['child_alone'], axis = 1, inplace = True)
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Message', engine, index=True, if_exists = 'replace')


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