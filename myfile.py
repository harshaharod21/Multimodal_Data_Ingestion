from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pandas as pd
import PyPDF2
import mailparser

def newFunction():

    text_path=["C:\All_projects\ImageBind\.assets\Text.txt"]
    image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
    audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]
    video_paths=["C:\All_projects\ImageBind\.assets\sample_video_01.mp4"]
    pdf_paths=["C:\All_projects\ImageBind\.assets\sample_pdf_01.pdf"]
    csv_paths=["C:\All_projects\ImageBind\.assets\sample_csv.csv"]
    email_paths=["C:\All_projects\ImageBind\.assets\Application for AI Researcher Role.eml"]


    columns = ['path','media_type','embeddings']
    df = pd.DataFrame(columns=columns)


    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    def getEmbeddingVector(inputs):
        with torch.no_grad():
            embedding = model(inputs)
        for key, value in embedding.items():
            vec = value.reshape(-1)
            vec = vec.numpy()
            return(vec)



    def dataToEmbedding(dataIn,dtype):
        if dtype == 'image':
            print("Image one")
            data_path = [dataIn]
            inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(data_path, device)
            }
        elif dtype in ('text', 'pdf', 'csv', 'email'):
            print("text one")
            txt = [dataIn]
            inputs = {
            ModalityType.TEXT: data.load_and_transform_text(txt, device)
            }
        
        elif dtype=='audio':
            print("audio one")
            aud= [dataIn]
            inputs= {
                ModalityType.AUDIO: data.load_and_transform_audio_data(aud, device)
            }
        elif dtype=='video':
            print('video one')
            vid=[dataIn]
            inputs={
                ModalityType.VISION: data.load_and_transform_video_data(vid,device)
            }

        vec = getEmbeddingVector(inputs)
        return(vec)
#abs
    for image in image_paths:
        path = image
        media_type = "image"
        embedding = dataToEmbedding(path,media_type)
        new_row = {'path': path,
                'media_type':media_type,
                'embeddings':embedding}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for text in text_path:
        path =  text
        media_type = "text"
        embedding = dataToEmbedding(path,media_type)
        new_row = {'path': path,
                'media_type':media_type,
                'embeddings':embedding}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for audio in audio_paths:
        path = audio
        media_type = "audio"
        embedding = dataToEmbedding(path,media_type)
        new_row = {'path': path,
                'media_type':media_type,
                'embeddings':embedding}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for video in video_paths:
        path = video
        media_type = "video"
        embedding = dataToEmbedding(path,media_type)
        new_row = {'path': path,
                'media_type':media_type,
                'embeddings':embedding}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for pdf in pdf_paths:
        path = pdf
        media_type = "pdf"
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text_pdf = ''
            for page in range(len(reader.pages)):
                text_pdf += reader.pages[page].extract_text()
                
        embedding = dataToEmbedding(text_pdf,media_type)
        new_row = {'path': path,
                'media_type':media_type,
                'embeddings':embedding}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for csv in csv_paths:
        path = csv
        media_type = "csv"
        df_csv = pd.read_csv(path)
        text_data = ' '.join(df_csv.apply(lambda row: ' '.join(row.values.astype(str)), axis=1))
        embedding = dataToEmbedding(text_data,media_type)
        new_row = {'path': path,
                'media_type':media_type,
                'embeddings':embedding}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for email in email_paths:
        path = email
        media_type = "email"
        parsed_mail = mailparser.parse_from_file(path)
        email_text = parsed_mail.body
        embedding = dataToEmbedding(email_text,media_type)
        new_row = {'path': path,
                'media_type':media_type,
                'embeddings':embedding}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)



    return df


   