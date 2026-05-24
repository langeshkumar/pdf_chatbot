import PyPDF2


def pdf_to_text(uploaded_files):
    
    combain_text = ""

    for pdf_file in uploaded_files:

        # pdf to pdf reader 
        text_reader = PyPDF2.PdfReader(pdf_file)

        # page data 
        for page in text_reader.pages:

            # page to text 
            pdf_text = page.extract_text()

            if(pdf_text):

                # combain all text 
                combain_text += pdf_text + "\n"

    return combain_text
