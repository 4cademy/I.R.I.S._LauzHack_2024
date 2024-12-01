# IRIS - Intelligent Recognition & Image Search

## Demo Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Ukx69JS2VaU/0.jpg)](https://www.youtube.com/watch?v=Ukx69JS2VaU)

## Prerequisites
```
git clone https://github.com/4cademy/I.R.I.S._LauzHack_2024.git repo
cd repo
touch .env
```
- Create an API-KEY for SerpApi here: https://serpapi.com/manage-api-key
- Past it into the created `.env` file like this: `SEARCH_API_KEY=your_api_key_here`

```bash
export AWS_DEFAULT_REGION="us-west-2"
export AWS_ACCESS_KEY_ID="your_aws_access_key_id"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_access_key"
export AWS_SESSION_TOKEN="your_aws_session_token"
export OPENAI_API_KEY="your_openai_api_key"
```
## Installation
```bash
pip install -r requirements.txt
python app.py
```

Go to the URL shown in the console (normally: http://127.0.0.1:7860/)

