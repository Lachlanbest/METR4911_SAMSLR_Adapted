import pandas as pd

# Load train .csv file - change according to analysed dataset
train = pd.read_csv('/home/Student/s4582342/train_labels.csv')

# Load val .csv file - change according to analysed dataset
val = pd.read_csv('/home/Student/s4582342/val_labels.csv')

# Concatenate the two DataFrames vertically
concat_file = pd.concat([train, val], axis=0)

# Save concat file to new .csv file
concat_file.to_csv('/home/Student/s4582342/train_val_labels.csv', index=False)
