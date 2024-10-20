# Multimodal Data Ingestion into Vector Database

## Overview
This project aims to create a multimodal data ingestion pipeline using **ImageBind** to generate multimodal embeddings, and **KDBAI Vector Database** for storing these embeddings. 

**Note**: We are not using any frameworks like LangChain or LlamaIndex for this project because, as of now, they do not support integration with ImageBind for handling different data types (such as PDFs, CSVs, or emails) directly as input and converting them to text.

# ImageBind


[[`Paper`](https://facebookresearch.github.io/ImageBind/paper)] [[`Blog`](https://ai.facebook.com/blog/imagebind-six-modalities-binding-ai/)] [[`Demo`](https://imagebind.metademolab.com/)] [[`Supplementary Video`](https://dl.fbaipublicfiles.com/imagebind/imagebind_video.mp4)] [[`BibTex`](#citing-imagebind)]

PyTorch implementation and pretrained models for ImageBind. For details, see the paper: **[ImageBind: One Embedding Space To Bind Them All](https://facebookresearch.github.io/ImageBind/paper)**.

ImageBind learns a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. It enables novel emergent applications ‘out-of-the-box’ including cross-modal retrieval, composing modalities with arithmetic, cross-modal detection and generation.



![ImageBind](https://user-images.githubusercontent.com/8495451/236859695-ffa13364-3e39-4d99-a8da-fbfab17f9a6b.gif)


# KDB.AI Vector Database

![KDB.AI Logo](https://kdb.ai/files/2024/01/kdbai-logo.svg)

The example [KDB.AI](https://kdb.ai) samples provided aim to demonstrate examples of the use of the KDB.AI vector database in a number of scenarios ranging from getting started guides to industry specific use-cases.

## KDB.AI Offerings

KDB.AI comes in two offerings:

1. [KDB.AI Cloud](https://trykdb.kx.com/kdbai/signup/) - For experimenting with smaller generative AI projects with a vector database in our cloud.
2. [KDB.AI Server](https://trykdb.kx.com/kdbaiserver/signup/) - For evaluating large scale generative AI applications on-premises or on your own cloud provider.

Depending on which you use, there will be different setup steps and connection details required.
You can signup at the links above and see the notebooks for connection inctructions.

## What is KDB.AI?

KDB.AI is a vector database with time-series capabilities that allows developers to build scalable, reliable, and real-time applications by providing advanced search, recommendation, and personalization for Generative AI applications. KDB.AI is a key component of full-stack Generative AI applications that use Retrieval Augmented Generation (RAG).

Built by KX, the creators of kdb+, KDB.AI provides users with the ability to combine unstructured vector embedding data with structured time-series datasets to allow for hybrid use-cases which benefit from the rigor of conventional time-series data analytics and the usage patterns provided by vector databases within the Generative AI space.


## What does KDB.AI support?

KDB.AI supports the following feature set:

- Multiple index types: Flat, qFlat, IVF, IVFPQ, HNSW and qHnsw.
- Multiple distance metrics: Euclidean, Inner-Product, Cosine.
- Top-N and metadata filtered retrieval
- Python and REST Interfaces

# Setup Instructions

### Step 1: Clone the Repository
First, clone the ImageBind repository from GitHub:

```bash
git clone https://github.com/facebookresearch/ImageBind.git
```

### Step 2: Create and Activate a Conda Environment

```bash
conda create --name imagebind python=3.10 -y
conda activate imagebind
```

For Windows users, you might need to install soundfile to handle reading/writing of audio files:

```bash
pip install soundfile
```

**Note:**
- There may be version conflicts with Numpy. To resolve this, install the following version of Numpy:
```bash
conda install numpy=1.24
```

- Since we are using pandas to create dataframes, install the following version to ensure compatibility with the Numpy version installed in the previous step:

```bash
pip install pandas==1.5.3
```

**The setup is done!**

Next steps include creating the files **myfile.py** and **insert_data.py**.



## Setup of Kdb.ai vector database:

This guide will help you set up a **KDB.AI Vector Database** to store and manage vector embeddings. If you do not already have a KDB.AI account, you can sign up for free at [KDB.AI](https://kdb.ai).

### Prerequisites
- A KDB.AI account
- Python environment with `kdbai_client` installed

You will need to connect to a KDB.AI session, either through the cloud (recommended) or another instance, using your KDB.AI API key and endpoint.

### Step 1: Install Required Dependencies

```bash
pip install kdbai_client
```
### Step 2: Set Up KDB.AI Endpoint and API Key

To connect to KDB.AI, you will need your **KDB.AI Cloud endpoint URL** and **API key**. These can either be provided manually or through environment variables.

You can set the following environment variables on your system for automatic use in the code:

- `KDBAI_ENDPOINT`: Your KDB.AI endpoint URL
- `KDBAI_API_KEY`: Your KDB.AI API key

To set these variables on your system, run the following commands (for Unix-based systems):

```bash
export KDBAI_ENDPOINT="your_kdbai_endpoint"
export KDBAI_API_KEY="your_kdbai_api_key"
```

### Step 3: Connecting to the Database

Once the session is created using the endpoint and API key, you can proceed with vector database operations such as creating tables, inserting data, and running queries.

```bash
session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)
```
**This establishes a connection to KDB.AI, allowing you to interact with the vector database!**

After successful setup of the connection with KDB.AI vector database(the steps are given below) you can run the file insert_data.py in your terminal:

```bash
python insert_data.py
```

# Workings of myfile.py and insert_data.py

Now lets explore the files 'myfile.py' and 'insert_data.py'

**Note:** The data files contain all the 10 files of different data types.

## 1) Explanation of myfile.py

This Python script uses the **ImageBind** model to generate multimodal embeddings for different media types, including images, audio, video, PDFs, CSVs, and emails. The embeddings are stored in a pandas DataFrame, which can be later used for storage in a vector database like KDBAI for further analysis.

### Code Explanation

- **Imports**: 
  - `imagebind` for generating embeddings from different modalities (text, images, audio, etc.).
  - `PyPDF2` for extracting text from PDF files.
  - `mailparser` for parsing email files.
  - `pandas` to store and manage the generated embeddings in a DataFrame.
  
- **Media Types**: 
  The script supports processing the following media types: 
  - **Text**
  - **Images**
  - **Audio**
  - **Video**
  - **PDF**
  - **CSV**
  - **Emails**

- **Model Setup**:
  The `imagebind_huge` model is instantiated and set to evaluation mode to generate embeddings from input data. The model runs on a GPU (if available) or CPU.

- **Embedding Generation**: 
  The function `dataToEmbedding` handles different media types and converts them into embedding vectors using the appropriate loading and transformation functions provided by `imagebind`.

- **Data Processing**:
  For each media type (images, text, audio, etc.), the script:
  1. Loads the media file.
  2. Converts it into embeddings using the `dataToEmbedding` function.
  3. Appends the file path, media type, and embeddings to a pandas DataFrame.

- **PDF and CSV Handling**:
  - PDFs are parsed using `PyPDF2` to extract text from each page.
  - CSVs are read and converted to text format by concatenating rows for embedding generation.

- **Email Handling**:
  Emails are parsed using `mailparser` to extract the email body for embedding.

- **Output**:
  The function `newFunction()` returns a pandas DataFrame containing the paths, media types, and their corresponding embeddings.

**Usage:**

The script allows for multimodal data processing and embedding generation. These embeddings can be stored in a vector database for further analysis, making it ideal for applications involving complex multimedia data.

## 2) Explanation of insert_data.py:

This Python script inserts the multimodal embeddings generated by the **ImageBind** model (from `myfile.py`) into a **KDB.AI** vector database. The embeddings are stored in a structured table, allowing for efficient retrieval and querying of vector data.

### Code Explanation

- **Imports**: 
  - `kdbai_client` to interact with the KDB.AI vector database.
  - `myfile.py` to use the `newFunction()` function that generates embeddings.
  - `dotenv` to securely load API keys and environment variables.

- **Data Ingestion**:
  The DataFrame `df` containing multimodal embeddings is created by calling `newFunction()` from `myfile.py`.

- **KDB.AI Setup**:
  The script loads the KDB.AI **endpoint** and **API key** from environment variables (`.env` file) using `load_dotenv()`.

- **Database Connection**:
  A session is established with the KDB.AI database using the provided endpoint and API key.

- **Table Schema**:
  The table schema is defined with three columns:
  - `path`: Stores the file path of the media.
  - `media_type`: Stores the type of media (image, text, audio, etc.).
  - `embeddings`: Stores the embedding vectors with 1024 dimensions, indexed using **cosine similarity (CS)**.

- **Table Creation**:
  The script ensures that any pre-existing table named `multi_modal_ImageBind` is dropped, then creates a new table using the specified schema.

- **Data Insertion**:
  The DataFrame `df` is split into batches of 2000 rows for efficient insertion into the table. The `tqdm` library is used to display a progress bar during the insertion process.

- **Querying the Database**:
  After inserting the data, the script queries the table to explore its contents.

**Usage:**
This script enables efficient storage and retrieval of multimodal embeddings in the **KDB.AI** vector database. It defines a custom schema for storing the embeddings, and the data can be queried or explored after insertion.












