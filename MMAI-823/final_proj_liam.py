import pandas as pd
import os

# Get a list of files and directories in the current directory
files_and_dirs = os.listdir()

# Print the list of files and directories
print(files_and_dirs)

# Load the pickle file into a DataFrame
df = pd.read_pickle('MMAI-823/2018-04-01.pkl')

# Display the column headings
print(df.columns)

unique_counts = df.nunique()

print(unique_counts)

df.to_csv('fraud.csv', index=False)
