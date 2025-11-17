import requests
import streamlit as st

st.title("Simple FastAPI application")

tab1, tab2, tab3 = st.tabs(['Image', 'Text', 'Table'])

def main():
    with tab1:
        image = st.file_uploader("Detect objects in image", type=['jpg', 'jpeg'])
        if st.button("Detect!") and image is not None:
            st.image(image)
            files = {"file": image.getvalue()}
            res = requests.post("http://89.169.168.148:8000/clf_image", files=files).json()
            st.write(f'Objects detected: {res["total_objects"]}')
            for det in res["detections"]:
                st.write(f'{det["class_name"]}: {det["confidence"]:.2f}')

    with tab2:
        txt = st.text_input('Analyze text sentiment')
        if st.button('Analyze'):
            text = {'text': txt}
            res = requests.post("http://89.169.168.148:8000/clf_text", json=text).json()
            st.write(f'Sentiment: {res["label"]}')
            st.write(f'Confidence: {res["prob"]:.2f}')
        
    with tab3: 
        st.write("Demo table data")
        with st.form("query_form"):
            feature1 = st.text_input("Feature 1", value="0.")
            feature2 = st.text_input("Feature 2", value='0.')
            submitted = st.form_submit_button("Predict!")
            if submitted:
                vector = {'feature1': feature1, 'feature2': feature2}
                res = requests.post("http://89.169.168.148:8000/clf_table", json=vector).json()
                st.write(f"Predicted: {res['prediction']}")

if __name__ == '__main__':
    main()