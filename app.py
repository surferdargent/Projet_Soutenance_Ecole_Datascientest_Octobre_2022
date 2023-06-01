import streamlit as st
import config
st.set_page_config(
    page_title=config.TITLE,
    #page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)


from collections import OrderedDict

from tabs import intro, second_tab, third_tab,fourth_tab




with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (second_tab.sidebar_name, second_tab),
        (third_tab.sidebar_name, third_tab),
        (fourth_tab.sidebar_name, fourth_tab)
        
    ]
)


def run():
    st.sidebar.image(
        
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        width=200
    )
    
    #st.write("Battre les Bookmakers Tennis ?")
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### l'Equipe:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()
    st.sidebar.markdown("Sous le tutorat de *Gaspard Grimm*")

if __name__ == "__main__":
    run()
