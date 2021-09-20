import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Candara'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.size'] = 12

from adjustText.adjustText import adjust_text


# Here we develop an OOPs oriented format in which we have various methods to operate on the csv file

class DataInput:
    def __init__(self, filepath_data):
        self.data = pd.read_csv(filepath_data, index_col='Rk')
        try:
            self.data['Player'] = self.data['Player'].apply(lambda x: ' '.join(x.split('\\')[1].split('-')))
            self.data['Comp'] = self.data['Comp'].apply(lambda x: ' '.join(x.split()[1:]))
            self.data['Nation'] = self.data['Nation'].apply(lambda x: x.split()[-1])
        except:
            print('Error in loading file:' + filepath_data)
        
        self.filtereddf = self.data.copy()

    # The below method shows the head of the data after a filtering option. We can control the number of
    # data rows by changing the no option

    def showdata(self, no=5):
        print(self.filtereddf.head(no))

    # The below method will filter the data based on min number of 90s played, team, league and nation
    # We can list the positions we want to remove by adding them as parameters after the min_90s parameter
    # To find the names of different leagues, teams, nations and positions, we have defined various methods later on

    def filterdata(self, min_90s=5, *pos_to_remove, team: object = False,
                   league: object = False, nation: object = False):
        notpos = []
        for pos in pos_to_remove:
            notpos.append(pos)
        self.filtereddf = self.filtereddf[self.filtereddf['Pos'].apply(lambda x: x not in notpos)]
        self.filtereddf = self.filtereddf[self.filtereddf['90s'] >= min_90s]
        if team:
            self.filtereddf = self.filtereddf[self.filtereddf['Squad'] == team]
        if league:
            self.filtereddf = self.filtereddf[self.filtereddf['Comp'] == league]
        if nation:
            self.filtereddf = self.filtereddf[self.filtereddf['Nation'] == nation]

    # This method lists the various positions which can be found in the original or filtered dataset
    # To find positions in filtered dataset make the filtered_dataset param = True
    def show_different_positions(self, filtered_dataset=False):
        if filtered_dataset:
            print(self.filtereddf['Pos'].unique())
        print(self.data['Pos'].unique())

    # This method shows us the different nations which can be found in the original or filtered dataset
    def show_different_nations(self, filtered_dataset=False):
        if filtered_dataset:
            print(self.filtereddf['Pos'].unique())
        print(self.data['Pos'].unique())

    # This method shows all the the columns or only the stats columns in the original dataset
    def show_relevant_columns(self, all_cols=False):
        if all_cols:
            print(list(self.data.columns))

        columns = list(self.data.columns)
        col_to_remove = ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born', '90s']
        for col in col_to_remove:
            if col in columns:
                columns.remove(col)
        print(columns)

    # This properly lists all the names of the leagues in the original dataset
    @property
    def show_leaguenames(self):
        print(self.data['Comp'].unique())

    # This methods shows all the teams present in a particular league
    def show_teams_in_league(self, leaguename):
        print(self.data[self.data['Comp'] == leaguename]['Squad'].unique())

    # This methods shows all the players present in a particular team
    def show_players_in_team(self, teamname):
        print(self.data[self.data['Squad'] == teamname][['Player','90s','Pos']].reset_index(drop=True))

    # This method saves the filtered df as a csv file on your system
    # The first parameters saves the file in that name(don't add the .csv)
    # The second parameter saves the file in the directory you choose(if not sure, leave as be, it will
    # save the csv in the same directory as this Python script)
    def save_new_csv(self, filename='new_csv', filepath=os.getcwd()):
        try:
            self.filtereddf.to_csv(filepath + '\\' + filename + '.csv', encoding='utf-8-sig')
        except:
            print('An error occurred in saving the file')
        else:
            print('File has been saved as {0} at {1}'.format(filename, filepath))

    # This is a plotting method
    # on_x refers to the x-axis parameter while #on_y refers to the y-axis parameter
    # per90 makes the graph a per90 graph(make sure no parameters are per90 among the two though)
    # no_of_players limits the graph the top n players in the two metrics
    # season changes the input of season to add to title
    # plot_on_page plots the graph(default in Jupyter Notebook)
    # custom_headers lets you make custom labels for x, y axes and title
    # size is the figure size(input as a tuple, first refers to x width, second to y width)
    # point_col, text_col and bg_color refer to the colour of the points, the text and background respectively
    # annotate tells whether you want to label all points or not
    # player_focus labels a particular player you want to focus on in the filtered dataframe, for this annotate has to be off
    # save saves the plot in a directory of your choice
    # add_xref, add_yred and add_refline adds reference lines according to slope needed
    # repel creates a ggrepel type feature which repels each label
    # Above code taken from https://github.com/Phlya/adjustText
    def plot(self, on_x, on_y, per90=False, no_of_players=False, season='20-21', plot_on_page=True,
             custom_headers=False, figure_size=(12, 8), point_col='red', text_col='black',
             bgcolor='white', annotate = True, player_focus = None, save=False, add_xref=False,
             add_yref=False, add_refline=False, repel=True):

        df = self.filtereddf.copy().reset_index(drop=True)
        x_axis, y_axis = on_x, on_y

        p90statement: str = ''

        if per90:
            p90statement += ' per 90'
            if (on_x[-1] != '%') and (on_x[-2:] != '90'):
                df[on_x] = df[on_x] / df['90s']
                x_axis += p90statement
            if (on_y[-1] != '%') and (on_y[-2:] != '90'):
                df[on_y] = df[on_y] / df['90s']
                y_axis += p90statement

        if annotate or not player_focus:
            if no_of_players:
                scaled = MinMaxScaler().fit_transform(df[[on_x, on_y]])
                df_scaled = []
                for i in range(scaled.shape[0]):
                    df_scaled.append(np.prod(scaled[i]))
                df['Custom'] = pd.Series(df_scaled)
                # df['Custom'] = df[on_x] * df[on_y]
                df = df.sort_values('Custom', ascending=False)[:no_of_players].reset_index(drop=True)
                df.drop('Custom', axis=1, inplace=True)


        league = ''
        nation = ''
        team = ''
        if (int(df['Comp'].nunique()) > 1) and (int(df['Squad'].nunique()) > 1):
            title = '{0} vs {1} top {2} players across Europe\'s top 5 leagues for {3}{4}'.format(on_x, on_y,
                                                                                                str(no_of_players),
                                                                                                str(season),
                                                                                                p90statement)
            if int(df['Nation'].nunique()) == 1:
                nation += df['Nation'].iloc[0] + ' nationality'
                title = '{0} vs {1} top {2} players across Europe\'s top 5 leagues of {3} for {4}{5}'.format(on_x, on_y,
                                                                                                            str(no_of_players), nation,
                                                                                                            str(season),p90statement)
        else:
            if int(df['Comp'].nunique()) == 1:
                league += df['Comp'].iloc[0]
            if int(df['Squad'].nunique()) == 1:
                team += df['Squad'].iloc[0]
                league = ''
            if int(df['Nation'].nunique()) == 1:
                nation += df['Nation'].iloc[0] + ' nationality'

            # title_dict = {'League': league, 'Team': team, 'Nation': nation}
            title_lst = [league, team, nation]
            title_lst = [name for name in title_lst if len(name) > 0]

            if len(title_lst) == 1:
                title_st = title_lst[0]
            else:
                title_st = ' and of '.join(title_lst)

            title = '{0} vs {1} top {2} players in {3} for {4}{5}'.format(on_x, on_y, str(no_of_players), title_st,
                                                                            season, p90statement)
       

        if custom_headers:
            title = input('Title header: ')
            x_axis = input('X axis title: ')
            y_axis = input('Y axis title: ')

        fig, ax = plt.subplots(figsize=figure_size)
        ax.patch.set_facecolor(bgcolor)
        fig.set_facecolor(bgcolor)

        x_dist = (df[on_x].max() - df[on_x].min()) / 50
        y_dist = (df[on_y].max() - df[on_y].min()) / 50
        size = 12

        # SCATTERPLOT
        alpha = 0.75 if len(df.index)<30 else 0.375
        p = sns.scatterplot(x=on_x, y=on_y, data=df, color=point_col, edgecolor='black', alpha=alpha)
        p.grid(True, axis='both', c='gray', ls='--', alpha=0.5)


        # TEXT ANNOTATIONS
        if annotate:
            if not repel:
                for line in range(df.shape[0]):
                    plt.text(df[on_x][line] - x_dist, df[on_y][line] + y_dist,
                             df['Player'][line],
                             fontdict=dict(color=text_col, size=size))
            else:
                texts = []
                for x, y, s in zip(df[on_x], df[on_y], df['Player']):
                    texts.append(plt.text(x, y, s,
                                          fontdict=dict(color=text_col, size=size)))
                adjust_text(texts, force_points=0.2, force_text=0.2,
                            expand_points=(1, 1), expand_text=(1, 1),
                            arrowprops=dict(arrowstyle="-", color='white', lw=0.5,alpha=0))
        
        else:
            if player_focus:
                df_player = df[df['Player']==player_focus]
                try:
                    ax.scatter(df_player[on_x], df_player[on_y], s=50, c='black', ec='black', zorder=2, alpha=0.8)
                    ax.text(df_player[on_x] - x_dist, df_player[on_y] + y_dist, df_player['Player'].values[0], size=14)
                except: pass

        # FOOTER
        fig.text(0.01,0.015, 'Made using FootballFbrefPlotting by Shreyas Khatri/@khatri_shreyas. '+
        'Code repo present at https://github.com/shreyas7kha/FootballFbrefPlotting.', 
        size=10, weight='bold')

        # SET TITLES AND AXES LABELS
        p.set_xlabel(x_axis, weight='heavy', family='Century Gothic', size=15, color=text_col)
        p.set_ylabel(y_axis, weight='heavy', family='Century Gothic', size=15, color=text_col)
        p.set_title(title.upper(), weight='heavy', family='Century Gothic', size=20, color=text_col)

        if add_xref:
            p.axvline(add_xref, ls='--')

        if add_yref:
            p.axhline(add_yref, ls='--')

        if add_refline:
            slope, intercept = add_refline

            def abline(slope, intercept):
                """Plot a line from slope and intercept"""
                axes = plt.gca()
                x_vals = np.array(axes.get_xlim())
                y_vals = intercept + slope * x_vals
                plt.plot(x_vals, y_vals, '--')

            abline(slope, intercept)

        if save:
            file_path = input('Choose filepath you want to save in or click enter if in same directory: ')
            filename = input('Save file as(don\'t add the .png): ')

            if len(file_path.split()) == 0:
                file_path = os.getcwd()

            if input('Do you want to save the image in an uncreated sub directory:').lower()[0] == 'y':
                sub_dir = input('Name of the sub directory: ')
                os.mkdir(file_path + '\\' + sub_dir)
                file_path = file_path + '\\' + sub_dir

            try:
                plt.savefig(file_path + '\\' + filename + '.png', facecolor=bgcolor ,dpi=300)
            except:
                print('An error occurred in saving the plot')
            else:
                print(f'File was saved succesfully at {file_path} with name {filename}')
        if plot_on_page:
            plt.show()


# Many a times we want to create a plot from two different csv files, the below class helps us do so
# TwoDataInput takes 2 csvs as input and merges them, all the methods remain similar to the previous class

# A preliminary function which combines both dfs and cleans the dataframe
def combine_and_cleancolumns(d1, d2):
    data = pd.merge(d1, d2, on=['Player','Squad'])
    columns = data.columns
    new_cols = []

    for col in columns:
        if (col[-2:] != '_x') and (col[-2:]!='_y'):
            new_cols.append(col)
        else:
            if col[:-2] not in new_cols:
                new_cols.append(col[:-2])
            else:
                data.drop(col, inplace=True, axis=1)
    data.columns = new_cols
    try:
        data['Player'] = data['Player'].apply(lambda x: ' '.join(x.split('\\')[1].split('-')))
        data['Comp'] = data['Comp'].apply(lambda x: ' '.join(x.split()[1:]))
        data['Nation'] = data['Nation'].apply(lambda x: x.split()[-1])
    except:
        print('Error in loading file at ' + d1 + ' and ' + d2)
    finally:
        return data


class TwoDataInput(DataInput):
    def __init__(self, filepath_data1, filepath_data2):
        super().__init__(filepath_data1)
        super().__init__(filepath_data2)
        self.d1 = pd.read_csv(filepath_data1, index_col='Rk')
        self.d2 = pd.read_csv(filepath_data2, index_col='Rk')
        self.data = combine_and_cleancolumns(self.d1, self.d2)
        self.filtereddf = self.data.copy()


# A function which can read data from a Fbref webpage
def readfromhtml(filepath):
    df = pd.read_html(filepath)[0]
    
    column_lst = list(df.columns)
    unique_col_names = []
    for col in column_lst:
        if col[1] not in unique_col_names:
            unique_col_names.append(col[1])
        else:
            unique_col_names.append(col[0]+' '+col[1])

    df.columns = unique_col_names
    df.drop(df[df['Player'] == 'Player'].index, inplace=True)
    df = df.fillna('0')
    df.set_index('Rk', drop=True, inplace=True)
    try:
        df['Comp'] = df['Comp'].apply(lambda x: ' '.join(x.split()[1:]))
        df['Nation'] = df['Nation'].astype(str)
        df['Nation'] = df['Nation'].apply(lambda x: x.split()[-1])
    except:
        print('Error in uploading file:' + filepath)
    finally:
        df = df.apply(pd.to_numeric, errors='ignore')
        return df


# Similar to the data input, this particular class takes information directly from
# Fbref's website based on url
class DataInputFromWebpage(DataInput):
    def __init__(self, filepath_data):
        self.data = readfromhtml(filepath_data)
        self.filtereddf = self.data.copy()


# This is the two class extension extended to both files from Fbref's website
class TwoDataInputFromWebpage(DataInput):
    def __init__(self, filepath_data1, filepath_data2):
        self.d1 = readfromhtml(filepath_data1)
        self.d2 = readfromhtml(filepath_data2)
        self.data = pd.merge(self.d1, self.d2)
        self.filtereddf = self.data.copy()


from bs4 import BeautifulSoup as soup


# If you want all data for the big 5 leagues, you just need to run this function with
# the filepath where you want to save all the files
def save_all_csvs(base_url='https://fbref.com/en/comps/Big5/Big-5-European-Leagues-Stats',
                  filepath=os.getcwd()):
    req = requests.get(base_url)
    parse_soup = soup(req.content, 'lxml')
    scripts = parse_soup.find_all('ul')
    url_list = scripts[4]
    urls = []
    for url in url_list.find_all('a', href=True):
        urls.append(url['href'])
    urls = [base_url[:17] + url for url in urls]
    for url in urls:
        df = readfromhtml(url)
        filename = url.split('/')[6]
        try:
            df.to_csv(filepath + '\\' + filename + '.csv', encoding='utf-8-sig')
        except:
            print('An error occurred in saving the file')
        else:
            print('File has been saved as {0} at {1}'.format(filename, filepath))
