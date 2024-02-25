import pandas as pd

df = pd.read_csv('/Users/freddiehurdwood/Desktop/Uni Work/ForCancer/freddie/HAM10000_metadata.csv')
word_to_number = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}

# Replace word values with numbers in a specific column (change 'column_name' to your column's name)
df['dx'] = df['dx'].map(word_to_number)

columns_to_delete = ['dx_type', 'age', 'sex', 'localization', 'lesion_id']
for column in columns_to_delete:
    del df[column]


# Save the modified dataframe to a new CSV file
df.to_csv('new_file.csv', index=False)



