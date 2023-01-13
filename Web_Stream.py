import streamlit as st
import pandas as pd
import textwrap
# Load and Cache the data
@st.cache(persist=True)
def getdata():
    games_df = pd.read_csv('datasets/Games_dataset.csv', index_col=0)
    similarity_df = pd.read_csv('datasets/sim_matrix.csv', index_col=0)
    return games_df, similarity_df

games_df, similarity_df = getdata()[0], getdata()[1]

# Sidebar
st.sidebar.markdown('Game Recommend System \n'
                    'is built by Nguyen Van Hung')
st.sidebar.image('images/banner.png', use_column_width=True)
st.sidebar.markdown('# Choose your game!')
st.sidebar.markdown('')
ph = st.sidebar.empty()
selected_game = ph.selectbox('Select one games '
                             'from the menu dropdown: (you can type it)',
                             [''] + games_df['Title'].to_list(), key='default',
                             format_func=lambda x: 'Select a game' if x == '' else x)

# Recommendations
if selected_game:

    link = 'https://en.wikipedia.org' + games_df[games_df.Title == selected_game].Link.values[0]

    # DF query
    matches = similarity_df[selected_game].sort_values()[1:6]
    matches = matches.index.tolist()
    matches = games_df.set_index('Title').loc[matches]
    matches.reset_index(inplace=True)

    # Results
    cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan']

    st.markdown("# The recommended games for [{}]({}) are:".format(selected_game, link))
    for idx, row in matches.iterrows():
        st.markdown('### {} - {}'.format(str(idx +1), row['Title']))
        st.markdown('{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 300)[0], row['Link']))
        st.table(pd.DataFrame(row[cols]).T)
        st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))

