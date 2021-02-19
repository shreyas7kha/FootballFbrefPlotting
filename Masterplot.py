import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataInput:
    data: object

    def __init__(self, filepath_data):
        self.data = pd.read_csv(filepath_data, index_col='Rk')
        self.filtereddf = self.data
        self.filtereddf['Player'] = self.filtereddf['Player'].apply(lambda x: ' '.join(x.split('\\')[1].split('-')))
        self.filtereddf['Comp'] = self.filtereddf['Comp'].apply(lambda x: ' '.join(x.split()[1:]))

    def showdata(self, no=5):
        print(self.filtereddf.head(no))

    def filterdata(self, min_90s: object = 5, *pos_to_remove, team: object = False, league: object = False):
        notpos = []
        for pos in pos_to_remove:
            notpos.append(pos)
        self.filtereddf = self.filtereddf[self.filtereddf['Pos'].apply(lambda x: x not in notpos)]
        self.filtereddf = self.filtereddf[self.filtereddf['90s'] >= min_90s]
        if team:
            self.filtereddf = self.filtereddf[self.filtereddf['Squad'] == team]
        if league:
            self.filtereddf = self.filtereddf[self.filtereddf['Comp'] == league]

    def show_different_positions(self):
        return list(self.data['Pos'].unique())

    def show_relevant_columns(self, all=False):
        if all:
            return list(self.data.columns)

        col = list(self.data.columns)
        while True:
            if col[0] == '90s':
                col.remove('90s')
                break
            col.pop(0)

        return col

    def show_leaguenames(self):
        return list(self.data['Comp'].unique())

    def show_teams_in_league(self,leaguename):
        return list(self.data[self.data['Comp']==leaguename]['Squad'].unique())

    def save_new_csv(self, filename='new_csv'):
        self.filtereddf.to_csv(filename + '.csv')

    def plot(self, on_x, on_y, per90=True, no_of_players=False, season='20-21', plot_on_page=True, custom_headers=False, size=(12,8), save=False):

        df = self.filtereddf.reset_index(drop=True)
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

        if int(df['Comp'].nunique())>1:
            title = on_x + ' vs ' + on_y + ' top ' + str(no_of_players) + ' players in Europes top 5 leagues for ' \
                + str(season) + p90statement
        else:
            league = df['Comp'].iloc[0]
            title = on_x + ' vs ' + on_y + ' top ' + str(no_of_players) + ' players in ' + league +\
                ' for ' + str(season) + p90statement
        x_axis = on_x + p90statement
        y_axis = on_y + p90statement

        if custom_headers:
            title = input('Title header: ')
            x_axis = input('X axis title: ')
            y_axis = input('Y axis title: ')

        plt.figure(figsize=size)

        if not per90:
            p = sns.scatterplot(x=on_x, y=on_y, data=df, size=3,
                                hue='Pos', legend=False)
            for line in range(df.shape[0]):
                plt.text(df[on_x][line] + 0.1, df[on_y][line],
                         df['Player'][line],
                         fontdict=dict(color='black', size=10))
            p.set(xlabel=x_axis, ylabel=y_axis)
            plt.title(title)

        if per90:
            p90statement += ' per90'
            p = sns.scatterplot(x=on_x , y= on_y,
                                data=df, size=3, hue='Pos', legend=False)
            for line in range(df.shape[0]):
                plt.text(df[on_x][line] + 0.05, df[on_y][line],
                         df['Player'][line],
                         fontdict=dict(color='black', size=8))
            p.set(xlabel=x_axis, ylabel=y_axis)
            plt.title(title)

        if save:
            filepath = input('Choose filepath you want to save in or click enter if in same directory: ')
            filename = input('Save file as(don\'t add the .png): ')
            plt.savefig(filepath+'\\'+filename+'.png')

        if plot_on_page:
            plt.show()