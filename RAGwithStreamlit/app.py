import streamlit as st
import response

st.write("Hello!")
chain = response.process()
query = st.text_input('Type in your query', '')
if query:
    result = chain.invoke(query)
    st.write(result)