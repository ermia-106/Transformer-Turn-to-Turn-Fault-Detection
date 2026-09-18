# Import Libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from Inference import predict_fault

# Page Configuration
st.set_page_config(page_title = 'Transformer Turn-to-Turn Fault Detection',page_icon = '⚡',layout = 'wide')

# Custom CSS
st.markdown(
    '''
    <style>
    .main-title {
        font-size : 42px;
        font-weight : 700;
        text-align : center;
        margin-bottom : 5px;
    }
    .subtitle {
        text-align : center;
        font-size : 18px;
        color : #777;
        margin-bottom : 35px;
    }
    .section-title {
        font-size : 25px;
        font-weight : 600;
        margin-top : 15px;
        margin-bottom : 15px;
    }
    .footer {
        text-align : center;
        padding : 30px 0 10px 0;
        color : #777;
        font-size : 14px;
    }
    .footer a {
        text-decoration : none;
        margin : 0 10px;
    }
    </style>
    ''',
    unsafe_allow_html = True)

# Header
st.markdown('<div class = "main-title">⚡ Transformer Turn-to-Turn Fault Detection</div>',unsafe_allow_html = True)
st.markdown('<div class = "subtitle">Turn-to-Turn Short-Circuit Fault Detection using MATLAB and XGBoost</div>',unsafe_allow_html = True)

# Input Mode
st.markdown('<div class = "section-title">🔧 Input Data</div>',unsafe_allow_html = True)
input_mode = st.radio('Select input method :',['Manual Input','CSV Upload'],horizontal = True)

# Storage for Inference Inputs
Ia = None
Ib = None
Ic = None
load_power = None

# Manual Input
if input_mode == 'Manual Input' :
    st.info('Enter exactly 80 samples for each phase current. The sampling frequency must be 2 kHz.')
    col1 , col2 , col3 = st.columns(3)
    with col1 :
        st.markdown('### Phase A — Ia (A)')
        Ia_text = st.text_area('Ia Samples',height = 200,placeholder = 'Example : 10 , 20 , 30 , ...',label_visibility = 'collapsed')
    with col2 :
        st.markdown('### Phase B — Ib (A)')
        Ib_text = st.text_area('Ib Samples',height = 200,placeholder = 'Example : -50 , -40 , -30 , ...',label_visibility = 'collapsed')
    with col3 :
        st.markdown('### Phase C — Ic (A)')
        Ic_text = st.text_area('Ic Samples',height = 200,placeholder = 'Example : 40 , 30 , 20 , ...',label_visibility = 'collapsed')
    load_power = st.number_input('⚡ Load Power (W)',min_value = 0.0,value = 5_000_000.0,step = 100_000.0,format = '%.0f')
    if Ia_text :
        try :
            Ia = np.array([float(x.strip()) for x in Ia_text.replace('\n',',').split(',') if x.strip()])
        except ValueError :
            st.error('Invalid Values in Ia.')
    if Ib_text :
        try :
            Ib = np.array([float(x.strip()) for x in Ib_text.replace('\n',',').split(',') if x.strip()])
        except ValueError :
            st.error('Invalid Values in Ib.')
    if Ic_text :
        try :
            Ic = np.array([float(x.strip()) for x in Ic_text.replace('\n',',').split(',') if x.strip()])
        except ValueError :
            st.error('Invalid Values in Ic.')
    if Ia is not None :
        st.caption(f'Ia : {len(Ia)} / 80 Samples')
    if Ib is not None :
        st.caption(f'Ib : {len(Ib)} / 80 Samples')
    if Ic is not None :
        st.caption(f'Ic : {len(Ic)} / 80 Samples')

# CSV Upload
else :
    st.info('Upload a CSV containing Ia, Ib, Ic and Load_Power. The file must contain exactly 80 samples.')
    sampling_frequency = 2000
    frequency = 50
    amplitude = 100
    sample_count = 80
    load_power_sample = 5_000_000
    time = np.arange(sample_count) / sampling_frequency
    Ia_sample = (amplitude * np.sin(2 * np.pi * frequency * time))
    Ib_sample = (amplitude * np.sin(2 * np.pi * frequency * time - 2 * np.pi / 3))
    Ic_sample = (amplitude * np.sin(2 * np.pi * frequency * time + 2 * np.pi / 3))
    sample_csv = pd.DataFrame({'Ia' : Ia_sample,'Ib' : Ib_sample,'Ic' : Ic_sample,'Load_Power' : load_power_sample})
    st.download_button(label = '📥 Download Sample CSV',data = sample_csv.to_csv(index = False),file_name = 'Transformer_Data_Sample.csv',mime = 'text/csv',use_container_width = True)
    uploaded_file = st.file_uploader('Upload CSV File',type = ['csv'])
    if uploaded_file is not None :
        try :
            dataset = pd.read_csv(uploaded_file)
            required_columns = ['Ia','Ib','Ic','Load_Power']
            missing_columns = [column for column in required_columns if column not in dataset.columns]
            if missing_columns :
                st.error(f'Missing Columns : {", ".join(missing_columns)}')
            elif len(dataset) != 80 :
                st.error(f'CSV must contain exactly 80 rows. 'f'Your file contains {len(dataset)} rows.')
            elif dataset['Load_Power'].nunique() != 1 :
                st.error('Load_Power must be constant for all 80 samples.')
            else :
                Ia = dataset['Ia'].to_numpy(dtype = float)
                Ib = dataset['Ib'].to_numpy(dtype = float)
                Ic = dataset['Ic'].to_numpy(dtype = float)
                load_power = float(dataset['Load_Power'].iloc[0])
                st.success('CSV Loaded Successfully.')
                with st.expander('👀 Preview CSV') :
                    st.dataframe(dataset.head(10),use_container_width = True)
                st.caption(f'Loaded 80 Samples | 'f'Load Power : {load_power:,.0f} W')
        except Exception as error :
            st.error(f'Couldn\'t Read CSV : {error}')

# Waveforms
if (Ia is not None and Ib is not None and Ic is not None and len(Ia) == 80 and len(Ib) == 80 and len(Ic) == 80) :
    st.markdown('<div class = "section-title">📈 Current Waveforms</div>',unsafe_allow_html = True)
    time_ms = np.arange(80) / 2000 * 1000
    col1 , col2 , col3 = st.columns(3)
    with col1 :
        fig_a , ax_a = plt.subplots()
        ax_a.plot(time_ms,Ia)
        ax_a.set_title('Phase A — Ia')
        ax_a.set_xlabel('Time (ms)')
        ax_a.set_ylabel('Current (A)')
        ax_a.grid(True,alpha = 0.3)
        st.pyplot(fig_a,use_container_width = True)
        plt.close(fig_a)
    with col2 :
        fig_b , ax_b = plt.subplots()
        ax_b.plot(time_ms,Ib)
        ax_b.set_title('Phase B — Ib')
        ax_b.set_xlabel('Time (ms)')
        ax_b.set_ylabel('Current (A)')
        ax_b.grid(True,alpha = 0.3)
        st.pyplot(fig_b,use_container_width = True)
        plt.close(fig_b)
    with col3 :
        fig_c , ax_c = plt.subplots()
        ax_c.plot(time_ms,Ic)
        ax_c.set_title('Phase C — Ic')
        ax_c.set_xlabel('Time (ms)')
        ax_c.set_ylabel('Current (A)')
        ax_c.grid(True,alpha = 0.3)
        st.pyplot(fig_c,use_container_width = True)
        plt.close(fig_c)

# Prediction
st.markdown('---')
predict_button = st.button('🔍 Predict Transformer Condition',use_container_width = True,type = 'primary')
if predict_button :
    if Ia is None or Ib is None or Ic is None :
        st.warning('Please provide Ia, Ib and Ic data first.')
    elif len(Ia) != 80 :
        st.error(f'Ia must contain exactly 80 samples. 'f'Current : {len(Ia)}')
    elif len(Ib) != 80 :
        st.error(f'Ib must contain exactly 80 samples. 'f'Current : {len(Ib)}')
    elif len(Ic) != 80 :
        st.error(f'Ic must contain exactly 80 samples. 'f'Current : {len(Ic)}')
    elif load_power is None :
        st.warning('Please provide Load Power.')
    else :
        with st.spinner('Running MATLAB engine and XGBoost model...') :
            try :
                result = predict_fault(Ia,Ib,Ic,load_power)
                st.session_state['prediction_result'] = result
            except Exception as error :
                st.error(f'Inference Failed : {error}')

# Result
if 'prediction_result' in st.session_state :
    result = st.session_state['prediction_result']
    st.markdown('---')
    st.markdown('<div class = "section-title">🎯 Prediction Result</div>',unsafe_allow_html = True)
    if result['prediction'] == 0 :
        with st.container(border = True) :
            st.success('🟢 HEALTHY')
            st.write('No turn-to-turn fault was detected by the model.')
    else :
        with st.container(border = True) :
            st.error('🔴 FAULT DETECTED')
            st.write('The model detected a possible turn-to-turn fault.')
    col1 , col2 = st.columns(2)
    with col1 :
        st.metric('🟢 Healthy Probability',f"{result['healthy_probability'] * 100:.2f} %")
    with col2 :
        st.metric('🔴 Fault Probability',f"{result['fault_probability'] * 100:.2f} %")
    st.markdown('---')
    st.markdown('<div class = "section-title">⚙️ System Information</div>',unsafe_allow_html = True)
    col1 , col2 , col3 , col4 = st.columns(4)
    with col1 :
        st.metric('Samples / Phase','80')
    with col2 :
        st.metric('Sampling Frequency','2 kHz')
    with col3 :
        st.metric('Window Duration','40 ms')
    with col4 :
        st.metric('Model','XGBoost')

# Footer
st.markdown('---')
st.markdown('<div style = "text-align: center;"> </div>',unsafe_allow_html = True)
st.markdown('<div style = "text-align: center;">👨‍💻 Developed by ermia-106</div>',unsafe_allow_html = True)
st.markdown('<div style = "text-align: center;"> </div>',unsafe_allow_html = True)
col1 , col2 , col3 = st.columns(3)
with col1 :
    st.link_button('🔗 LinkedIn','https://www.linkedin.com/in/ermia106',use_container_width = True)
with col2 :
    st.link_button('💻 GitHub','https://github.com/ermia-106',use_container_width = True)
with col3 :
    st.link_button('✉️ Email','mailto:ermia.sh.106@gmail.com',use_container_width = True)
st.markdown('<div style = "text-align: center;">© 2026 • Built with Python, MATLAB & XGBoost</div>',unsafe_allow_html = True)