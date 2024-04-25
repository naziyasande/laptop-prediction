import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor ")

Company = st.selectbox('Brand', df['Company'].unique())
laptop_type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 32, 64])
weight = st.number_input('Weight of Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('Ips', ['No', 'Yes'])
screen_size = st.number_input('Screen size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3200x1800',
                                                '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Encoding categorical variables
    query = np.array([Company, type, cpu, gpu, os])
    query_encoded = np.zeros((1, len(df.columns)))
    for i, col in enumerate(df.columns):
        if col == 'Company':
            query_encoded[0, i] = np.where(df['Company'].unique() == Company)[0][0]
        elif col == 'TypeName':
            query_encoded[0, i] = np.where(df['TypeName'].unique() == type)[0][0]
        elif col == 'Cpu brand':
            query_encoded[0, i] = np.where(df['Cpu brand'].unique() == cpu)[0][0]
        elif col == 'Gpu brand':
            query_encoded[0, i] = np.where(df['Gpu brand'].unique() == gpu)[0][0]
        elif col == 'os':
            query_encoded[0, i] = np.where(df['os'].unique() == os)[0][0]
        elif col == 'Ram':
            query_encoded[0, i] = ram
        elif col == 'Weight':
            query_encoded[0, i] = weight
        elif col == 'Touchscreen':
            query_encoded[0, i] = touchscreen
        elif col == 'IPS':
            query_encoded[0, i] = ips
        elif col == 'ppi':
            query_encoded[0, i] = ppi
        elif col == 'HDD':
            query_encoded[0, i] = hdd
        elif col == 'SSD':
            query_encoded[0, i] = ssd
        elif col == 'ScreenSize':
            query_encoded[0, i] = screen_size

    price = pipe.predict(query_encoded)
    st.title("The Estimated Price of the Laptop in INR: Rs {}".format(int(np.exp(price[0]))))
