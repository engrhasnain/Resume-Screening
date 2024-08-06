import streamlit as st
import pickle
import re

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# web app
def main():
    st.title("Resume Screening Application")
    upload_file = st.file_uploader('Upload Resume', type= ['txt', 'pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        clean_resume = cleanResume(resume_text)
        input_feature = tfidf.transform([clean_resume])
        prediction_id = clf.predict(input_feature)[0]

        cat = {
            6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Desinging',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Develpor',
            3: 'Blockchain',
            23: 'Testing'

        }

        category_name = cat.get(prediction_id, "Unknown")
        st.write("Predicted Category:" , category_name)


#python main
if __name__ == "__main__":
    main()
