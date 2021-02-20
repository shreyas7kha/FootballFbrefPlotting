# A sample code file for people not familiar with Python or object-oriented programming
# Save the csv files and both the Masterplot and Code file py scripts within the same folder on your system
# If not, change the value of cd to the directory where the particular csv file is kept

import os
from Masterplot import DataInput, TwoDataInput

cd = os.getcwd()

# The first example deals with the first class using only one csv file
# We take the data from https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats
# We filter data using parameters min 5 90s played, removing GK positions, and playing in the
# Premier League with Eng nationality
# Then we plot the graph on the KP and PPA columns, on a per 90 basis, and take the top 15 players
# The legend is shown in the graph
# We also save the graph in a directory you choose and with your own name

df = DataInput(cd + "\\Passing.csv")

df.filterdata(5, 'GK', league='Premier League', nation='ENG')
df.plot('KP', 'PPA', per90=True, no_of_players=15, show_legend=True)

# This example deals with two csv data sources which we want to merge
# We initialize an instance df2 of the two csv class
# Data for Passing csv - https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats
# Data for SCA csv - https://fbref.com/en/comps/Big5/gca/players/Big-5-European-Leagues-Stats
# We filter the players with min 10 90s, and remove any player with position GK or DF
# Then we call the showdata method which will show the filtered dataframe
# We then invoke the plot method again based on the GCA90 column from the SCA csv and 1/3 column
# from the Passing csv file and restrict it to the top 20 players

df2 = TwoDataInput(cd + "\\Passing.csv", cd + "\\SCA.csv")
df2.filterdata(10, 'GK', 'DF')
df2.showdata()
df2.plot('GCA90', '1/3', no_of_players=20)