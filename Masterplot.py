import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from adjustText.adjustText import adjust_text


# Here we develop an OOPs oriented format in which we have various methods to play with the csv file

class DataInput:
    def __init__(self, filepath_data):
        self.data = pd.read_csv(filepath_data, index_col='Rk')
        self.data['Player'] = self.data['Player'].apply(lambda x: ' '.join(x.split('\\')[1].split('-')))
        self.data['Comp'] = self.data['Comp'].apply(lambda x: ' '.join(x.split()[1:]))
        self.data['Nation'] = self.data['Nation'].apply(lambda x: x.split()[1])
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
            return self.filtereddf['Pos'].unique()
        return self.data['Pos'].unique()

    # This method shows us the different nations which can be found in the original or filtered dataset
    def show_different_nations(self, filtered_dataset=False):
        if filtered_dataset:
            return self.filtereddf['Pos'].unique()
        return self.data['Nation'].unique()

    # This method shows all the the columns or only the stats columns in the original dataset
    def show_relevant_columns(self, all_cols=False):
        if all_cols:
            return list(self.data.columns)

        columns = list(self.data.columns)
        col_to_remove = ['Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born', '90s']
        for col in col_to_remove:
            if col in columns:
                columns.remove(col)
        return columns

    # This properly lists all the names of the leagues in the original dataset
    @property
    def show_leaguenames(self):
        return self.data['Comp'].unique()

    # This methods shows all the teams present in a particular league
    def show_teams_in_league(self, leaguename):
        return self.data[self.data['Comp'] == leaguename]['Squad'].unique()

    # This method saves the filtered df as a csv file on your system
    # The first parameters saves the file in that name(don't add the .csv)
    # The second parameter saves the file in the directory you choose(if not sure, leave as be, it will
    # save the csv in the same directory as this Python script)
    def save_new_csv(self, filename='new_csv', filepath=os.getcwd()):
        try:
            self.filtereddf.to_csv(filepath + '\\' + filename + '.csv')
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
    # save saves the plot in a directory of your choice
    # change_hue changes the hue of the plot, it is by default based on Position
    # show_legend shows the legend on the plot
    # add_xref, add_yred and add_refline adds reference lines according to slope needed
    # repel creates a ggrepel type feature which repels each label
    # Above code taken from https://github.com/Phlya/adjustText
    def plot(self, on_x, on_y, per90=False, no_of_players=False, season='20-21', plot_on_page=True,
             custom_headers=False, size=(12, 8), save=False, change_hue='Pos', show_legend=False, add_xref=False,
             add_yref=False, add_refline=False, repel=True):

        df = self.filtereddf.copy().reset_index(drop=True)
        if on_x[-2:] == '90' or on_y[-2:] == 90:
            per90 = False

        if per90:
            df[on_x] = df[on_x] / df['90s']
            df[on_y] = df[on_y] / df['90s']

        if no_of_players:
            number = no_of_players
            df['Custom'] = df[on_x] * df[on_y]
            df = df.sort_values('Custom', ascending=False)[:number].reset_index(drop=True)

        p90statement: str = ''
        if per90:
            p90statement += ' per 90'

        league = ''
        nation = ''
        team = ''
        if (int(df['Comp'].nunique()) > 1) and (int(df['Squad'].nunique()) > 1):
            title = '{0} vs {1} top {2} players across Europe\'s top 5 leagues\n for {3}{4}'.format(on_x, on_y,
                                                                                                    str(no_of_players),
                                                                                                    str(season),
                                                                                                    p90statement)
            if int(df['Nation'].nunique()) == 1:
                nation += df['Nation'].iloc[0] + ' nationality'
                title = '{0} vs {1} top {2} players across Europe\'s top 5 leagues of {3} \n for {4}{5}'.format(on_x,
                                                                                                                on_y,
                                                                                                                str(
                                                                                                                    no_of_players),
                                                                                                                nation,
                                                                                                                str(
                                                                                                                    season),
                                                                                                                p90statement)
        else:
            if int(df['Comp'].nunique()) == 1:
                league += df['Comp'].iloc[0]
            if int(df['Squad'].nunique()) == 1:
                team += df['Squad'].iloc[0]
                league = ''
            if int(df['Nation'].nunique()) == 1:
                nation += df['Nation'].iloc[0] + ' nationality'

            title_dict = {'League': league, 'Team': team, 'Nation': nation}
            title_lst = list(title_dict.values())
            title_lst = [name for name in title_lst if len(name) > 0]

            if len(title_lst) == 1:
                title_st = title_lst[0]
            else:
                title_st = ' and of '.join(title_lst)

            title = '{0} vs {1} top {2} players in {3}\n for {4}{5}'.format(on_x, on_y, str(no_of_players), title_st,
                                                                            season, p90statement)
        x_axis = on_x + p90statement
        y_axis = on_y + p90statement

        if custom_headers:
            title = input('Title header: ')
            x_axis = input('X axis title: ')
            y_axis = input('Y axis title: ')

        plt.figure(figsize=size)
        plt.style.use('ggplot')

        dist = (df[on_x].max() - df[on_x].min()) / 100
        size = [8 if per90 else 10][0]

        p = sns.scatterplot(x=on_x, y=on_y, hue=change_hue,
                            data=df, legend=show_legend)
        if not repel:
            for line in range(df.shape[0]):
                plt.text(df[on_x][line] + dist, df[on_y][line],
                        df['Player'][line],
                        fontdict=dict(color='black', size=size))
        else:
            texts = []
            for x,y,s in zip(np.array(df[on_x]), df[on_y], df['Player']):
                texts.append(plt.text(x, y, s,
                                      fontdict=dict(color='black', size=size)))
            adjust_text(texts, force_points=0.2, force_text=0.2,
                        expand_points=(1, 1), expand_text=(1, 1),
                        arrowprops=dict(arrowstyle="-", color='white', lw=0.5))
        p.set(xlabel=x_axis, ylabel=y_axis)
        plt.title(title)

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
                os.mkdir(file_path+'\\'+sub_dir)
                file_path = file_path+'\\'+sub_dir


            try:
                plt.savefig(file_path + '\\' + filename + '.png')
            except:
                print('An error occurred in saving the plot')
            else:
                print(f'File was saved succesfully at {file_path} with name {filename}')
        if plot_on_page:
            plt.show()


# Many a times we want to create a plot from two different csv files, the below class helps us do so
# TwoDataInput takes 2 csvs as input and merges them, all the methods remain similar to the previous class

class TwoDataInput(DataInput):
    def __init__(self, filepath_data1, filepath_data2):
        super().__init__(filepath_data1)
        super().__init__(filepath_data2)
        self.d1 = pd.read_csv(filepath_data1, index_col='Rk')
        self.d2 = pd.read_csv(filepath_data2, index_col='Rk')
        self.data = pd.merge(self.d1, self.d2, on='Player')
        col = list(self.data.columns)
        new_col = []
        for i in col:
            name = i
            if i[-2:] == '_x' or i[-2:] == '_y':
                name = i[:-2]
            if name in new_col:
                self.data = self.data.drop(i, axis=1)
            else:
                new_col.append(name)
        self.data.columns = new_col
        self.data['Player'] = self.data['Player'].apply(lambda x: ' '.join(x.split('\\')[1].split('-')))
        self.data['Comp'] = self.data['Comp'].apply(lambda x: ' '.join(x.split()[1:]))
        self.data['Nation'] = self.data['Nation'].apply(lambda x: x.split()[1])
        self.filtereddf = self.data.copy()
