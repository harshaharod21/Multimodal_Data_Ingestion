# vector DB imports
import os
from getpass import getpass
import kdbai_client as kdbai
import time
from myfile import newFunction
from dotenv import load_dotenv

load_dotenv()

df=newFunction()

# Set up KDB.AI endpoint and API key
KDBAI_ENDPOINT = os.getenv("KDBAI_ENDPOINT") 
KDBAI_API_KEY = os.getenv("KDBAI_API_KEY") 

#connect to KDB.AI
session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)

#Define the schema

table_schema = {
    "columns": [
        {"name": "path", "pytype": "str"},
        {"name": "media_type", "pytype": "str"},
        {
            "name": "embeddings",
            "pytype": "float64",
            "vectorIndex": {"dims": 1024, "metric": "CS", "type": "flat"},
        },
    ]
}

# First ensure the table does not already exist
try:
    session.table("multi_modal_ImageBind").drop()
    time.sleep(5)
except kdbai.KDBAIException:
    pass

#Create the table called "multi_modal_demo"
table = session.create_table("multi_modal_ImageBind", table_schema)


#Insert the data into the table, split into 2000 row batches
from tqdm import tqdm 
n = 2000  # chunk row size

for i in tqdm(range(0, df.shape[0], n)):
    table.insert(df[i:i+n].reset_index(drop=True))


#Explore what is in the table/vector database
print(table.query())




