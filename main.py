import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

features = [
    'Q4_-_In_which_country_do_you_currently_reside?',
    'Q27_-_Does_your_current_employer_incorporate_machine_learning_methods_into_their_business?',
    'Q16_-_For_how_many_years_have_you_used_machine_learning_methods?',
    'Q11_-_For_how_many_years_have_you_been_writing_code_and/or_programming?',
    'Q23_-_Select_the_title_most_similar_to_your_current_role_(or_most_recent_title_if_retired):_-_Selected_Choice',
    'Q24_-_In_what_industry_is_your_current_employer/contract_(or_your_most_recent_employer_if_retired)?_-_Selected_Choice',
    'Q2_-_What_is_your_age_(#_years)?',
    'Q26_-_Approximately_how_many_individuals_are_responsible_for_data_science_workloads_at_your_place_of_business?',
    'Q25_-_What_is_the_size_of_the_company_where_you_are_employed?',
    'Q9_-_Have_you_ever_published_any_academic_research_(papers,_preprints,_conference_proceedings,_etc)?',
    'Q43_-_Approximately_how_many_times_have_you_used_a_TPU_(tensor_processing_unit)?', 
    'Q12_1_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Python',
    'Q12_2_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_R',
    'Q12_3_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_SQL',
    'Q12_4_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_C',
    'Q12_5_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_C#',
    'Q12_6_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_C++',
    'Q12_7_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Java',
    'Q12_8_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Javascript',
    'Q12_9_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Bash',
    'Q12_10_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_PHP',
    'Q12_11_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_MATLAB',
    'Q12_12_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Julia',
    'Q12_13_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Go',
    'Q12_14_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_None',
    'Q12_15_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Other'
]

# Load the model and label encoders
if "data_loaded" not in st.session_state:
    # Loading Dataset
    with st.spinner('Model Deployment in Progress. Please wait.'):
        df = pd.read_csv("BUS458_FinalCase.csv")
        df.drop(columns=['Duration (in seconds) - Duration (in seconds)', 'Q29 ', ' What is your current yearly compensation (approximate $USD)?'], inplace=True)
        df.columns=[col.replace(' ', '_') for col in df.columns]
        df[features[:12]]=df[features[:12]].fillna('None')
        df[features[12:]]=df[features[12:]].fillna('No')
        df.dropna(subset='Q29_-_Mean_Salary', inplace=True)
        
        options={}
        for col in features:
            options[col] = df[col].unique().tolist()
        label_encoders = {}
        for col in features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        
        X = df[features]
        y = df['Q29_-_Mean_Salary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier(n_estimators=20, 
                                    max_depth=10, 
                                    learning_rate=0.144, 
                                    min_samples_split=3, 
                                    random_state=42)
        model.fit(X_train, y_train)
        st.session_state.data_loaded = True
        st.session_state.model=model
        st.session_state.label_encoders=label_encoders
        st.session_state.options = options





st.title('Machine Learning Model Predictions')
st.write('This app generates predictions using a regression model.')
input_dict={}
#create int input
st.write('Input Features')
st.write('Select the values for the following features:')

if "input_dict" not in st.session_state:
    with st.form(key='my_form'):
        input_dict['Q4_-_In_which_country_do_you_currently_reside?'] =  st.session_state.label_encoders["Q4_-_In_which_country_do_you_currently_reside?"].transform([st.selectbox('Q4', st.session_state.options['Q4_-_In_which_country_do_you_currently_reside?'])])
        input_dict['Q27_-_Does_your_current_employer_incorporate_machine_learning_methods_into_their_business?'] = st.session_state.label_encoders["Q27_-_Does_your_current_employer_incorporate_machine_learning_methods_into_their_business?"].transform([st.selectbox('Q27', st.session_state.options['Q27_-_Does_your_current_employer_incorporate_machine_learning_methods_into_their_business?'])])
        input_dict['Q16_-_For_how_many_years_have_you_used_machine_learning_methods?'] = st.session_state.label_encoders["Q16_-_For_how_many_years_have_you_used_machine_learning_methods?"].transform([st.selectbox('Q16', st.session_state.options['Q16_-_For_how_many_years_have_you_used_machine_learning_methods?'])])
        input_dict['Q11_-_For_how_many_years_have_you_been_writing_code_and/or_programming?'] = st.session_state.label_encoders["Q11_-_For_how_many_years_have_you_been_writing_code_and/or_programming?"].transform([st.selectbox('Q11', st.session_state.options['Q11_-_For_how_many_years_have_you_been_writing_code_and/or_programming?'])])
        input_dict['Q23_-_Select_the_title_most_similar_to_your_current_role_(or_most_recent_title_if_retired):_-_Selected_Choice'] = st.session_state.label_encoders["Q23_-_Select_the_title_most_similar_to_your_current_role_(or_most_recent_title_if_retired):_-_Selected_Choice"].transform([st.selectbox('Q23', st.session_state.options['Q23_-_Select_the_title_most_similar_to_your_current_role_(or_most_recent_title_if_retired):_-_Selected_Choice'])])
        input_dict['Q24_-_In_what_industry_is_your_current_employer/contract_(or_your_most_recent_employer_if_retired)?_-_Selected_Choice'] = st.session_state.label_encoders["Q24_-_In_what_industry_is_your_current_employer/contract_(or_your_most_recent_employer_if_retired)?_-_Selected_Choice"].transform([st.selectbox('Q24', st.session_state.options['Q24_-_In_what_industry_is_your_current_employer/contract_(or_your_most_recent_employer_if_retired)?_-_Selected_Choice'])])
        input_dict['Q2_-_What_is_your_age_(#_years)?'] = st.session_state.label_encoders["Q2_-_What_is_your_age_(#_years)?"].transform([st.selectbox('Q2', st.session_state.options['Q2_-_What_is_your_age_(#_years)?'])])
        input_dict['Q26_-_Approximately_how_many_individuals_are_responsible_for_data_science_workloads_at_your_place_of_business?'] = st.session_state.label_encoders["Q26_-_Approximately_how_many_individuals_are_responsible_for_data_science_workloads_at_your_place_of_business?"].transform([st.selectbox('Q26', st.session_state.options['Q26_-_Approximately_how_many_individuals_are_responsible_for_data_science_workloads_at_your_place_of_business?'])])
        input_dict['Q25_-_What_is_the_size_of_the_company_where_you_are_employed?'] = st.session_state.label_encoders["Q25_-_What_is_the_size_of_the_company_where_you_are_employed?"].transform([st.selectbox('Q25', st.session_state.options['Q25_-_What_is_the_size_of_the_company_where_you_are_employed?'])])
        input_dict['Q9_-_Have_you_ever_published_any_academic_research_(papers,_preprints,_conference_proceedings,_etc)?'] = st.session_state.label_encoders["Q9_-_Have_you_ever_published_any_academic_research_(papers,_preprints,_conference_proceedings,_etc)?"].transform([st.selectbox('Q9', st.session_state.options['Q9_-_Have_you_ever_published_any_academic_research_(papers,_preprints,_conference_proceedings,_etc)?'])])
        input_dict['Q43_-_Approximately_how_many_times_have_you_used_a_TPU_(tensor_processing_unit)?'] = st.session_state.label_encoders["Q43_-_Approximately_how_many_times_have_you_used_a_TPU_(tensor_processing_unit)?"].transform([st.selectbox('Q43', st.session_state.options['Q43_-_Approximately_how_many_times_have_you_used_a_TPU_(tensor_processing_unit)?'])])

        input_dict['Q12_1_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Python'] = 0
        input_dict['Q12_2_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_R'] = 0
        input_dict['Q12_3_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_SQL'] = 0
        input_dict['Q12_4_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_C'] = 0
        input_dict['Q12_5_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_C#'] = 0
        input_dict['Q12_6_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_C++'] = 0
        input_dict['Q12_7_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Java'] = 0
        input_dict['Q12_8_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Javascript'] = 0
        input_dict['Q12_9_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Bash'] = 0
        input_dict['Q12_10_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_PHP'] = 0
        input_dict['Q12_11_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_MATLAB'] = 0
        input_dict['Q12_12_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Julia'] = 0
        input_dict['Q12_13_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Go'] = 0
        input_dict['Q12_14_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_None'] = 0
        input_dict['Q12_15_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Other'] = 0

        Q12_1 = st.checkbox('Python')
        if Q12_1:
            input_dict['Q12_1_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Python'] = 1
        Q12_2 = st.checkbox('R')
        if Q12_2:
            input_dict['Q12_2_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_R'] = 1

        Q12_3 = st.checkbox('SQL')
        if Q12_3:
            input_dict['Q12_3_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_SQL'] = 1
        Q12_4 = st.checkbox('C')
        if Q12_4:
            input_dict['Q12_4_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_C'] = 1
        Q12_5 = st.checkbox('C#')
        if Q12_5:
            input_dict['Q12_5_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_C#'] = 1
        Q12_6 = st.checkbox('C++')
        if Q12_6:
            input_dict['Q12_6_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_C++'] = 1
        Q12_7 = st.checkbox('Java')
        if Q12_7:
            input_dict['Q12_7_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Java'] = 1
        Q12_8 = st.checkbox('Javascript')
        if Q12_8:
            input_dict['Q12_8_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Javascript'] = 1
        Q12_9 = st.checkbox('Bash')
        if Q12_9:
            input_dict['Q12_9_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Bash'] = 1
        Q12_10 = st.checkbox('PHP')
        if Q12_10:
            input_dict['Q12_10_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_PHP'] = 1
        Q12_11 = st.checkbox('MATLAB')
        if Q12_11:
            input_dict['Q12_11_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_MATLAB'] = 1
        Q12_12 = st.checkbox('Julia')
        if Q12_12:
            input_dict['Q12_12_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Julia'] = 1
        Q12_13 = st.checkbox('Go')
        if Q12_13:
            input_dict['Q12_13_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Go'] = 1
        Q12_14 = st.checkbox('None')
        if Q12_14:
            input_dict['Q12_14_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_None'] = 1
        Q12_15 = st.checkbox('Other')
        if Q12_15:
            input_dict['Q12_15_-_What_programming_languages_do_you_use_on_a_regular_basis?_(Select_all_that_apply)_-_Selected_Choice_-_Other'] = 1

        submit = st.form_submit_button(label='Submit')
        if submit:
            st.session_state.input_dict = input_dict


if submit:
    # Read data file
    data = pd.DataFrame(st.session_state.input_dict, index=[0])
    st.write(data.columns)
    data=data[features]
    # Generate predictions
    predictions = st.session_state.model.predict(data)
    st.write('Predictions:')
    st.write(pd.DataFrame({'Prediction': predictions}))

