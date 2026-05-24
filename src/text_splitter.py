from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_splitterchunk(document):
    
    if(document):

        # split text and overlap 
        split_text = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        # text split 
        docvalue = split_text.split_text(document)

        return docvalue